import { useState, useEffect } from 'react';
import { storage } from '../lib/storage';

/**
 * React hook for using localStorage with React state
 * Automatically syncs state with localStorage
 */
export function useLocalStorage<T>(
  key: string,
  initialValue: T
): [T, (value: T | ((prev: T) => T)) => void, () => void] {
  // Initialize state from localStorage or use initial value
  const [storedValue, setStoredValue] = useState<T>(() => {
    const item = storage.get<T>(key);
    return item !== null ? item : initialValue;
  });

  // Update localStorage when state changes
  useEffect(() => {
    storage.set(key, storedValue);
  }, [key, storedValue]);

  // Setter function that works like useState
  const setValue = (value: T | ((prev: T) => T)) => {
    setStoredValue((prev) => {
      const newValue = value instanceof Function ? value(prev) : value;
      return newValue;
    });
  };

  // Function to remove the value from localStorage and reset to initial
  const removeValue = () => {
    storage.remove(key);
    setStoredValue(initialValue);
  };

  return [storedValue, setValue, removeValue];
}
