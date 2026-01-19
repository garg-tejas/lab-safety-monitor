/**
 * Safe localStorage utilities with versioning and error handling.
 * Handles quota exceeded, disabled storage, and incognito mode gracefully.
 */

const VERSION = "v1";

/**
 * Storage keys with versioning
 */
export const STORAGE_KEYS = {
  SIDEBAR_COLLAPSED: `sentinelvision:sidebar-collapsed:${VERSION}`,
  THEME: `sentinelvision:theme:${VERSION}`,
  SETTINGS: `sentinelvision:settings:${VERSION}`,
} as const;

/**
 * Safely get an item from localStorage
 */
export function getStorageItem(key: string): string | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    return localStorage.getItem(key);
  } catch (error) {
    console.warn(`Failed to read from localStorage (key: ${key}):`, error);
    return null;
  }
}

/**
 * Safely set an item in localStorage
 */
export function setStorageItem(key: string, value: string): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  try {
    localStorage.setItem(key, value);
    return true;
  } catch (error) {
    if (error instanceof DOMException) {
      // Handle quota exceeded
      if (error.name === "QuotaExceededError") {
        console.error("localStorage quota exceeded. Consider clearing old data.");
      }
      // Handle storage disabled (incognito, security settings)
      else if (error.name === "SecurityError") {
        console.warn("localStorage is disabled (incognito mode or security settings)");
      }
    }
    console.warn(`Failed to write to localStorage (key: ${key}):`, error);
    return false;
  }
}

/**
 * Safely remove an item from localStorage
 */
export function removeStorageItem(key: string): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  try {
    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.warn(`Failed to remove from localStorage (key: ${key}):`, error);
    return false;
  }
}

/**
 * Safely get a JSON object from localStorage
 */
export function getStorageJSON<T>(key: string, defaultValue: T): T {
  const item = getStorageItem(key);
  if (item === null) {
    return defaultValue;
  }

  try {
    return JSON.parse(item) as T;
  } catch (error) {
    console.warn(`Failed to parse JSON from localStorage (key: ${key}):`, error);
    return defaultValue;
  }
}

/**
 * Safely set a JSON object in localStorage
 */
export function setStorageJSON<T>(key: string, value: T): boolean {
  try {
    const json = JSON.stringify(value);
    return setStorageItem(key, json);
  } catch (error) {
    console.warn(`Failed to stringify JSON for localStorage (key: ${key}):`, error);
    return false;
  }
}

/**
 * Migrate old storage keys to new versioned keys
 */
export function migrateStorageKeys(): void {
  if (typeof window === "undefined") {
    return;
  }

  const migrations: Array<{ old: string; new: string }> = [
    { old: "marketwise-sidebar-collapsed", new: STORAGE_KEYS.SIDEBAR_COLLAPSED },
    { old: "marketwise-theme", new: STORAGE_KEYS.THEME },
    { old: "marketwise-settings", new: STORAGE_KEYS.SETTINGS },
    { old: "sentinelvision-sidebar-collapsed", new: STORAGE_KEYS.SIDEBAR_COLLAPSED },
    { old: "sentinelvision-theme", new: STORAGE_KEYS.THEME },
    { old: "sentinelvision-settings", new: STORAGE_KEYS.SETTINGS },
  ];

  for (const { old, new: newKey } of migrations) {
    const oldValue = getStorageItem(old);
    if (oldValue !== null) {
      setStorageItem(newKey, oldValue);
      removeStorageItem(old);
    }
  }
}
