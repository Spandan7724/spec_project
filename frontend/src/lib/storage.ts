/**
 * Type-safe localStorage utility functions
 */

export const storage = {
  /**
   * Get an item from localStorage with JSON parsing
   */
  get: <T>(key: string, defaultValue?: T): T | null => {
    try {
      const item = window.localStorage.getItem(key);
      if (item === null) {
        return defaultValue ?? null;
      }
      return JSON.parse(item) as T;
    } catch (error) {
      console.error(`Error reading from localStorage key "${key}":`, error);
      return defaultValue ?? null;
    }
  },

  /**
   * Set an item in localStorage with JSON serialization
   */
  set: <T>(key: string, value: T): void => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Error writing to localStorage key "${key}":`, error);
    }
  },

  /**
   * Remove an item from localStorage
   */
  remove: (key: string): void => {
    try {
      window.localStorage.removeItem(key);
    } catch (error) {
      console.error(`Error removing localStorage key "${key}":`, error);
    }
  },

  /**
   * Clear all items from localStorage
   */
  clear: (): void => {
    try {
      window.localStorage.clear();
    } catch (error) {
      console.error('Error clearing localStorage:', error);
    }
  },

  /**
   * Check if a key exists in localStorage
   */
  has: (key: string): boolean => {
    return window.localStorage.getItem(key) !== null;
  },
};

/**
 * Storage keys used throughout the application
 */
export const STORAGE_KEYS = {
  THEME: 'theme',
  CHAT_SESSIONS: 'chat_sessions',
  ANALYSIS_HISTORY: 'analysis_history',
  USER_PREFERENCES: 'user_preferences',
  ACTIVE_SESSION: 'active_session',
} as const;
