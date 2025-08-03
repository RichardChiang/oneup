"""
Lichess tactics data download and processing pipeline.

Downloads, processes, and loads Lichess puzzle database into PostgreSQL.
Implements data quality filtering and unicode conversion.
"""

import asyncio
import bz2
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiofiles
import aiohttp
import asyncpg
from tqdm.asyncio import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.chess import ChessConverter, ChessConversionError
from backend.database import initialize_database, get_session_scope
from backend.database.models import TacticsPuzzle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LichessDataPipeline:
    """
    Pipeline for downloading and processing Lichess tactics data.
    
    Handles download, decompression, parsing, validation, and database loading.
    """
    
    def __init__(
        self,
        data_dir: str = "data/lichess",
        database_url: Optional[str] = None,
        min_rating: int = 1000,
        max_rating: int = 2500,
        max_puzzles: Optional[int] = None
    ):
        """
        Initialize data pipeline.
        
        Args:
            data_dir: Directory to store downloaded data
            database_url: Database connection URL
            min_rating: Minimum puzzle rating to include
            max_rating: Maximum puzzle rating to include
            max_puzzles: Maximum number of puzzles to process (None for all)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.max_puzzles = max_puzzles
        
        # File paths
        self.compressed_file = self.data_dir / "lichess_db_puzzle.csv.bz2"
        self.csv_file = self.data_dir / "lichess_db_puzzle.csv"
        self.processed_file = self.data_dir / "processed_puzzles.jsonl"
        
        # Download URL
        self.download_url = "https://database.lichess.org/lichess_db_puzzle.csv.bz2"
        
        # Statistics
        self.stats = {
            "downloaded": 0,
            "parsed": 0,
            "valid": 0,
            "invalid": 0,
            "inserted": 0,
            "errors": 0
        }
    
    async def run_full_pipeline(self):
        """Run the complete data pipeline."""
        logger.info("Starting Lichess data pipeline...")
        
        try:
            # Step 1: Download data
            await self.download_data()
            
            # Step 2: Decompress
            await self.decompress_data()
            
            # Step 3: Process and validate
            await self.process_puzzles()
            
            # Step 4: Load to database
            await self.load_to_database()
            
            # Step 5: Create indexes
            await self.create_indexes()
            
            # Print final statistics
            self.print_statistics()
            
            logger.info("Lichess data pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    async def download_data(self) -> bool:
        """
        Download Lichess puzzle database.
        
        Returns:
            True if download successful or file already exists
        """
        if self.compressed_file.exists():
            logger.info(f"Compressed file already exists: {self.compressed_file}")
            return True
        
        logger.info(f"Downloading Lichess puzzle database from {self.download_url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.download_url) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Download failed with status {response.status}")
                    
                    total_size = int(response.headers.get('content-length', 0))
                    
                    # Create progress bar
                    progress = tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc="Downloading"
                    )
                    
                    async with aiofiles.open(self.compressed_file, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                            progress.update(len(chunk))
                    
                    progress.close()
            
            self.stats["downloaded"] = 1
            logger.info(f"Download completed: {self.compressed_file}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if self.compressed_file.exists():
                self.compressed_file.unlink()
            raise
    
    async def decompress_data(self) -> bool:
        """
        Decompress the downloaded bz2 file.
        
        Returns:
            True if decompression successful or file already exists
        """
        if self.csv_file.exists():
            logger.info(f"CSV file already exists: {self.csv_file}")
            return True
        
        logger.info("Decompressing Lichess puzzle database...")
        
        try:
            # Get compressed file size for progress
            compressed_size = self.compressed_file.stat().st_size
            
            with bz2.open(self.compressed_file, 'rt', encoding='utf-8') as compressed:
                with open(self.csv_file, 'w', encoding='utf-8') as decompressed:
                    # Create progress bar
                    progress = tqdm(
                        total=compressed_size,
                        unit='B',
                        unit_scale=True,
                        desc="Decompressing"
                    )
                    
                    chunk_size = 8192
                    while True:
                        chunk = compressed.read(chunk_size)
                        if not chunk:
                            break
                        
                        decompressed.write(chunk)
                        progress.update(len(chunk.encode('utf-8')))
                    
                    progress.close()
            
            logger.info(f"Decompression completed: {self.csv_file}")
            return True
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            if self.csv_file.exists():
                self.csv_file.unlink()
            raise
    
    async def process_puzzles(self) -> bool:
        """
        Process and validate puzzle data.
        
        Returns:
            True if processing successful
        """
        logger.info("Processing and validating puzzles...")
        
        try:
            # Count total lines for progress
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f) - 1  # Subtract header
            
            valid_puzzles = []
            
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Create progress bar
                progress = tqdm(
                    total=min(total_lines, self.max_puzzles or total_lines),
                    desc="Processing puzzles"
                )
                
                for i, row in enumerate(reader):
                    if self.max_puzzles and i >= self.max_puzzles:
                        break
                    
                    try:
                        puzzle = await self.process_single_puzzle(row)
                        if puzzle:
                            valid_puzzles.append(puzzle)
                            self.stats["valid"] += 1
                        else:
                            self.stats["invalid"] += 1
                        
                        self.stats["parsed"] += 1
                        
                        # Save in batches to avoid memory issues
                        if len(valid_puzzles) >= 1000:
                            await self.save_batch(valid_puzzles)
                            valid_puzzles = []
                        
                    except Exception as e:
                        logger.warning(f"Error processing puzzle {i}: {e}")
                        self.stats["errors"] += 1
                    
                    progress.update(1)
                
                progress.close()
                
                # Save remaining puzzles
                if valid_puzzles:
                    await self.save_batch(valid_puzzles)
            
            logger.info(f"Processing completed. Valid: {self.stats['valid']}, Invalid: {self.stats['invalid']}")
            return True
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    async def process_single_puzzle(self, row: Dict[str, str]) -> Optional[Dict]:
        """
        Process and validate a single puzzle.
        
        Args:
            row: CSV row dictionary
            
        Returns:
            Processed puzzle dictionary or None if invalid
        """
        try:
            # Extract fields
            puzzle_id = row.get('PuzzleId', '').strip()
            fen = row.get('FEN', '').strip()
            moves = row.get('Moves', '').strip()
            rating = row.get('Rating', '0')
            rd = row.get('RatingDeviation', '0')
            popularity = row.get('Popularity', '0')
            nb_plays = row.get('NbPlays', '0')
            themes = row.get('Themes', '').strip()
            game_url = row.get('GameUrl', '').strip()
            opening_tags = row.get('OpeningTags', '').strip()
            
            # Validate required fields
            if not puzzle_id or not fen or not moves:
                return None
            
            # Parse numeric fields
            try:
                rating = int(rating)
                rd = int(rd) if rd else None
                popularity = int(popularity) if popularity else 0
                nb_plays = int(nb_plays) if nb_plays else 0
            except ValueError:
                return None
            
            # Filter by rating
            if rating < self.min_rating or rating > self.max_rating:
                return None
            
            # Validate FEN and convert to unicode
            try:
                unicode_position = ChessConverter.fen_to_unicode(fen)
                if not ChessConverter.validate_unicode(unicode_position):
                    return None
            except ChessConversionError:
                return None
            
            # Parse themes
            themes_list = []
            if themes:
                themes_list = [theme.strip() for theme in themes.split() if theme.strip()]
            
            # Parse opening tags
            opening_list = []
            if opening_tags:
                opening_list = [tag.strip() for tag in opening_tags.split() if tag.strip()]
            
            # Calculate difficulty level (1-5) based on rating
            if rating < 1200:
                difficulty_level = 1
            elif rating < 1500:
                difficulty_level = 2
            elif rating < 1800:
                difficulty_level = 3
            elif rating < 2100:
                difficulty_level = 4
            else:
                difficulty_level = 5
            
            # Count moves in solution
            move_count = len(moves.split()) if moves else 0
            
            return {
                'id': puzzle_id,
                'fen': fen,
                'moves': moves,
                'rating': rating,
                'rd': rd,
                'popularity': popularity,
                'nb_plays': nb_plays,
                'themes': themes_list,
                'game_url': game_url if game_url else None,
                'opening_tags': opening_list,
                'unicode_position': unicode_position,
                'difficulty_level': difficulty_level,
                'move_count': move_count
            }
            
        except Exception as e:
            logger.warning(f"Failed to process puzzle: {e}")
            return None
    
    async def save_batch(self, puzzles: List[Dict]):
        """Save a batch of puzzles to JSONL file."""
        async with aiofiles.open(self.processed_file, 'a', encoding='utf-8') as f:
            for puzzle in puzzles:
                await f.write(json.dumps(puzzle) + '\n')
    
    async def load_to_database(self) -> bool:
        """
        Load processed puzzles to database.
        
        Returns:
            True if loading successful
        """
        logger.info("Loading puzzles to database...")
        
        try:
            # Initialize database
            db = initialize_database(self.database_url)
            
            # Create tables if they don't exist
            db.create_tables()
            
            # Count lines for progress
            with open(self.processed_file, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            
            # Load in batches
            batch_size = 100
            batch = []
            
            progress = tqdm(total=total_lines, desc="Loading to database")
            
            async with get_session_scope() as session:
                async with aiofiles.open(self.processed_file, 'r', encoding='utf-8') as f:
                    async for line in f:
                        try:
                            puzzle_data = json.loads(line.strip())
                            
                            # Create TacticsPuzzle object
                            puzzle = TacticsPuzzle(**puzzle_data)
                            batch.append(puzzle)
                            
                            # Insert batch when full
                            if len(batch) >= batch_size:
                                await self.insert_batch(session, batch)
                                self.stats["inserted"] += len(batch)
                                batch = []
                            
                            progress.update(1)
                            
                        except Exception as e:
                            logger.warning(f"Failed to load puzzle: {e}")
                            self.stats["errors"] += 1
                
                # Insert remaining puzzles
                if batch:
                    await self.insert_batch(session, batch)
                    self.stats["inserted"] += len(batch)
            
            progress.close()
            
            logger.info(f"Database loading completed. Inserted: {self.stats['inserted']} puzzles")
            return True
            
        except Exception as e:
            logger.error(f"Database loading failed: {e}")
            raise
    
    async def insert_batch(self, session, puzzles: List[TacticsPuzzle]):
        """Insert a batch of puzzles using bulk insert."""
        try:
            session.add_all(puzzles)
            await session.flush()
        except Exception as e:
            # Handle conflicts by inserting individually
            for puzzle in puzzles:
                try:
                    session.add(puzzle)
                    await session.flush()
                except Exception:
                    await session.rollback()
    
    async def create_indexes(self) -> bool:
        """Create database indexes for optimal query performance."""
        logger.info("Creating database indexes...")
        
        try:
            # Connect directly to database for index creation
            conn = await asyncpg.connect(self.database_url)
            
            indexes = [
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tactics_rating ON tactics_puzzles(rating);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tactics_themes_gin ON tactics_puzzles USING GIN(themes);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tactics_difficulty ON tactics_puzzles(difficulty_level);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tactics_popularity ON tactics_puzzles(popularity DESC);",
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tactics_composite ON tactics_puzzles(difficulty_level, rating, popularity DESC);",
            ]
            
            for index_sql in indexes:
                try:
                    await conn.execute(index_sql)
                    logger.info(f"Created index: {index_sql.split()[-1]}")
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")
            
            await conn.close()
            
            logger.info("Database indexes created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            return False
    
    def print_statistics(self):
        """Print pipeline statistics."""
        logger.info("=" * 50)
        logger.info("LICHESS DATA PIPELINE STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Downloaded files: {self.stats['downloaded']}")
        logger.info(f"Puzzles parsed: {self.stats['parsed']:,}")
        logger.info(f"Valid puzzles: {self.stats['valid']:,}")
        logger.info(f"Invalid puzzles: {self.stats['invalid']:,}")
        logger.info(f"Puzzles inserted: {self.stats['inserted']:,}")
        logger.info(f"Errors encountered: {self.stats['errors']:,}")
        
        if self.stats['parsed'] > 0:
            success_rate = (self.stats['valid'] / self.stats['parsed']) * 100
            logger.info(f"Success rate: {success_rate:.2f}%")
        
        logger.info("=" * 50)


async def main():
    """Main entry point for the data pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process Lichess tactics data")
    parser.add_argument("--data-dir", default="data/lichess", help="Data directory")
    parser.add_argument("--database-url", help="Database URL (from env if not specified)")
    parser.add_argument("--min-rating", type=int, default=1000, help="Minimum puzzle rating")
    parser.add_argument("--max-rating", type=int, default=2500, help="Maximum puzzle rating")
    parser.add_argument("--max-puzzles", type=int, help="Maximum puzzles to process")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    parser.add_argument("--skip-processing", action="store_true", help="Skip processing step")
    parser.add_argument("--only-database", action="store_true", help="Only load to database")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = LichessDataPipeline(
        data_dir=args.data_dir,
        database_url=args.database_url,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        max_puzzles=args.max_puzzles
    )
    
    try:
        if args.only_database:
            await pipeline.load_to_database()
            await pipeline.create_indexes()
        elif args.skip_download and args.skip_processing:
            await pipeline.load_to_database()
            await pipeline.create_indexes()
        elif args.skip_download:
            await pipeline.decompress_data()
            await pipeline.process_puzzles()
            await pipeline.load_to_database()
            await pipeline.create_indexes()
        elif args.skip_processing:
            await pipeline.download_data()
            await pipeline.decompress_data()
            await pipeline.load_to_database()
            await pipeline.create_indexes()
        else:
            await pipeline.run_full_pipeline()
        
        pipeline.print_statistics()
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())