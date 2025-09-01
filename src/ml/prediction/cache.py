"""
Prediction caching system
"""

import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict
import logging

from .types import MLPredictionResponse

logger = logging.getLogger(__name__)


class PredictionCache:
    """
    In-memory cache for ML predictions with TTL support
    """
    
    def __init__(self, default_ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl_seconds
        
    def _get_cache_key(self, currency_pair: str, horizons: List[int]) -> str:
        """Generate cache key for prediction"""
        horizons_str = "_".join(map(str, sorted(horizons)))
        return f"{currency_pair}_{horizons_str}"
    
    def store_prediction(self, 
                        currency_pair: str,
                        prediction: MLPredictionResponse,
                        ttl_seconds: Optional[int] = None) -> None:
        """
        Store prediction in cache
        
        Args:
            currency_pair: Currency pair identifier
            prediction: Prediction response to cache
            ttl_seconds: Time to live in seconds (uses default if None)
        """
        ttl = ttl_seconds or self.default_ttl
        expiry_time = time.time() + ttl
        
        # Get horizons from prediction
        horizons = [int(h.replace('d', '')) for h in prediction.predictions.keys()]
        cache_key = self._get_cache_key(currency_pair, horizons)
        
        # Convert prediction to dict for storage
        prediction_dict = asdict(prediction)
        
        self.cache[cache_key] = {
            'prediction': prediction_dict,
            'stored_at': time.time(),
            'expires_at': expiry_time,
            'currency_pair': currency_pair,
            'horizons': horizons
        }
        
        logger.debug(f"Cached prediction for {currency_pair}, expires in {ttl}s")
    
    def get_prediction(self,
                      currency_pair: str,
                      horizons: List[int],
                      max_age_hours: int = 1) -> Optional[MLPredictionResponse]:
        """
        Get cached prediction if available and not expired
        
        Args:
            currency_pair: Currency pair identifier
            horizons: Prediction horizons requested
            max_age_hours: Maximum age in hours to consider valid
            
        Returns:
            Cached prediction or None if not available/expired
        """
        cache_key = self._get_cache_key(currency_pair, horizons)
        
        if cache_key not in self.cache:
            return None
        
        cached_item = self.cache[cache_key]
        current_time = time.time()
        
        # Check expiry
        if current_time > cached_item['expires_at']:
            del self.cache[cache_key]
            logger.debug(f"Expired cache entry removed for {currency_pair}")
            return None
        
        # Check age
        age_seconds = current_time - cached_item['stored_at']
        max_age_seconds = max_age_hours * 3600
        
        if age_seconds > max_age_seconds:
            logger.debug(f"Cache entry too old for {currency_pair}: {age_seconds}s")
            return None
        
        # Convert back to response object
        prediction_dict = cached_item['prediction']
        
        try:
            prediction = MLPredictionResponse(**prediction_dict)
            logger.debug(f"Retrieved cached prediction for {currency_pair}")
            return prediction
        except Exception as e:
            logger.error(f"Error deserializing cached prediction: {e}")
            del self.cache[cache_key]
            return None
    
    def invalidate_prediction(self, currency_pair: str, horizons: List[int] = None):
        """
        Invalidate cached predictions for a currency pair
        
        Args:
            currency_pair: Currency pair to invalidate
            horizons: Specific horizons to invalidate (all if None)
        """
        if horizons is not None:
            # Invalidate specific horizons
            cache_key = self._get_cache_key(currency_pair, horizons)
            if cache_key in self.cache:
                del self.cache[cache_key]
                logger.info(f"Invalidated cache for {currency_pair}, horizons: {horizons}")
        else:
            # Invalidate all for this currency pair
            keys_to_remove = []
            for cache_key, cached_item in self.cache.items():
                if cached_item['currency_pair'] == currency_pair:
                    keys_to_remove.append(cache_key)
            
            for key in keys_to_remove:
                del self.cache[key]
            
            logger.info(f"Invalidated all cached predictions for {currency_pair}")
    
    def cleanup_old_predictions(self, max_age_hours: int = 24):
        """
        Remove old cached predictions
        
        Args:
            max_age_hours: Maximum age in hours before removal
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        keys_to_remove = []
        
        for cache_key, cached_item in self.cache.items():
            age_seconds = current_time - cached_item['stored_at']
            
            # Remove if expired or too old
            if (current_time > cached_item['expires_at'] or 
                age_seconds > max_age_seconds):
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        if keys_to_remove:
            logger.info(f"Cleaned up {len(keys_to_remove)} old cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        
        total_entries = len(self.cache)
        expired_entries = 0
        currency_counts = {}
        
        for cached_item in self.cache.values():
            # Count expired
            if current_time > cached_item['expires_at']:
                expired_entries += 1
            
            # Count by currency pair
            currency_pair = cached_item['currency_pair']
            currency_counts[currency_pair] = currency_counts.get(currency_pair, 0) + 1
        
        # Calculate memory usage estimate (rough)
        memory_usage_bytes = 0
        for cached_item in self.cache.values():
            try:
                item_json = json.dumps(cached_item)
                memory_usage_bytes += len(item_json.encode('utf-8'))
            except:
                memory_usage_bytes += 1024  # Rough estimate
        
        return {
            'total_entries': total_entries,
            'expired_entries': expired_entries,
            'active_entries': total_entries - expired_entries,
            'entries_by_currency': currency_counts,
            'memory_usage_mb': memory_usage_bytes / (1024 * 1024),
            'default_ttl_seconds': self.default_ttl
        }
    
    def clear_cache(self):
        """Clear all cached predictions"""
        entries_cleared = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {entries_cleared} cache entries")
    
    def export_cache(self) -> Dict[str, Any]:
        """Export cache contents for debugging/analysis"""
        current_time = time.time()
        
        export_data = {
            'export_timestamp': current_time,
            'export_datetime': datetime.fromtimestamp(current_time).isoformat(),
            'cache_entries': []
        }
        
        for cache_key, cached_item in self.cache.items():
            entry_data = cached_item.copy()
            entry_data['cache_key'] = cache_key
            entry_data['is_expired'] = current_time > cached_item['expires_at']
            entry_data['age_seconds'] = current_time - cached_item['stored_at']
            
            export_data['cache_entries'].append(entry_data)
        
        return export_data
    
    def load_cache_from_export(self, export_data: Dict[str, Any]):
        """Load cache from exported data"""
        current_time = time.time()
        loaded_count = 0
        
        for entry_data in export_data.get('cache_entries', []):
            # Skip expired entries
            if current_time > entry_data.get('expires_at', 0):
                continue
            
            cache_key = entry_data['cache_key']
            
            # Remove cache-specific fields
            cached_item = entry_data.copy()
            for field in ['cache_key', 'is_expired', 'age_seconds']:
                cached_item.pop(field, None)
            
            self.cache[cache_key] = cached_item
            loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} cache entries from export")
    
    def get_prediction_history(self, currency_pair: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get history of cached predictions for a currency pair
        Note: This is limited to currently cached items
        """
        history = []
        
        for cached_item in self.cache.values():
            if cached_item['currency_pair'] == currency_pair:
                prediction_dict = cached_item['prediction']
                
                history_item = {
                    'timestamp': prediction_dict.get('timestamp'),
                    'model_confidence': prediction_dict.get('model_confidence'),
                    'horizons': cached_item['horizons'],
                    'cached_at': datetime.fromtimestamp(cached_item['stored_at']).isoformat(),
                    'processing_time_ms': prediction_dict.get('processing_time_ms')
                }
                
                history.append(history_item)
        
        # Sort by timestamp
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return history[:limit]