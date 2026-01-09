/**
 * Environment Configuration
 * Centralizes all environment-dependent values
 */

/**
 * Get the API URL from environment variables
 * Falls back to localhost for development
 */
export const getApiUrl = (): string => {
  return import.meta.env.VITE_API_URL || 'http://localhost:8000';
};

/**
 * Get the default complexity threshold
 * Can be overridden via environment variable
 */
export const getDefaultComplexityThreshold = (): number => {
  const envValue = import.meta.env.VITE_COMPLEXITY_THRESHOLD;
  if (envValue) {
    const parsed = parseFloat(envValue);
    if (!isNaN(parsed) && parsed >= 0 && parsed <= 1) {
      return parsed;
    }
  }
  return 0.8;
};

/**
 * Check if we're in development mode
 */
export const isDevelopment = (): boolean => {
  return import.meta.env.DEV;
};

/**
 * Check if we're in production mode
 */
export const isProduction = (): boolean => {
  return import.meta.env.PROD;
};
