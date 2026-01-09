/**
 * API Service for Modal Backend
 * Handles document extraction requests with response validation
 */

import { getApiUrl, getDefaultComplexityThreshold } from '../config/environment';
import {
  parseExtractionResult,
  ExtractionResult,
} from '../schemas/validation';
import { z } from 'zod';

// Get API URL from environment config
const MODAL_API_URL = getApiUrl();

// Default request timeout (5 minutes for long extractions)
const DEFAULT_TIMEOUT_MS = 300000;

export interface ExtractionRequest {
  file: File;
  complexityThreshold?: number;
  forceSystem2?: boolean;
}

export interface BatchExtractionRequest {
  files: File[];
  complexityThreshold?: number;
  forceSystem2?: boolean;
  enableVerification?: boolean;
}

// Custom error types for better error handling
export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public errorType: 'network' | 'validation' | 'server' | 'timeout' | 'unknown' = 'unknown'
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export class ValidationError extends Error {
  public errors: z.ZodError['errors'];
  constructor(message: string, errors: z.ZodError['errors']) {
    super(message);
    this.name = 'ValidationError';
    this.errors = errors;
  }
}

/**
 * Create an AbortController with timeout
 */
function createTimeoutController(timeoutMs: number): { controller: AbortController; timeoutId: ReturnType<typeof setTimeout> } {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  return { controller, timeoutId };
}

/**
 * Handle HTTP response errors with specific error types
 */
async function handleResponseError(response: Response): Promise<never> {
  const errorText = await response.text().catch(() => 'Unknown error');

  if (response.status === 400) {
    throw new ApiError(`Invalid request: ${errorText}`, response.status, 'validation');
  }
  if (response.status === 429) {
    throw new ApiError('Too many requests. Please try again later.', response.status, 'server');
  }
  if (response.status >= 500) {
    throw new ApiError(`Server error: ${errorText}`, response.status, 'server');
  }

  throw new ApiError(`Request failed: ${errorText}`, response.status, 'unknown');
}

/**
 * Validate batch extraction response
 */
function validateBatchResponse(data: unknown): ExtractionResult[] {
  if (!Array.isArray(data)) {
    throw new ValidationError('Expected array response', []);
  }

  const results: ExtractionResult[] = [];

  data.forEach((item, index) => {
    const parsed = parseExtractionResult(item);
    if (parsed.valid && parsed.data) {
      results.push(parsed.data);
    } else {
      // Log warning but don't fail entire batch
      console.warn(`Result ${index} validation failed: ${parsed.error}`);
      // Include result anyway for partial success
      results.push(item as ExtractionResult);
    }
  });

  return results;
}

export const extractDocument = async (
  request: ExtractionRequest,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<ExtractionResult> => {
  const { controller, timeoutId } = createTimeoutController(timeoutMs);

  try {
    const formData = new FormData();
    formData.append('file', request.file);
    formData.append('complexity_threshold', String(request.complexityThreshold ?? getDefaultComplexityThreshold()));
    formData.append('force_system2', String(request.forceSystem2 ?? false));

    const response = await fetch(`${MODAL_API_URL}/extract`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      await handleResponseError(response);
    }

    const data = await response.json();

    // Validate response
    const parsed = parseExtractionResult(data);
    if (!parsed.valid) {
      console.warn('Response validation warning:', parsed.error);
      // Return data anyway for graceful degradation
      return data as ExtractionResult;
    }

    return parsed.data!;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError('Request timed out', 0, 'timeout');
    }
    if (error instanceof ApiError || error instanceof ValidationError) {
      throw error;
    }
    throw new ApiError(
      error instanceof Error ? error.message : 'Unknown error',
      0,
      'network'
    );
  } finally {
    clearTimeout(timeoutId);
  }
};

export const extractBatch = async (
  request: BatchExtractionRequest,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<ExtractionResult[]> => {
  const { controller, timeoutId } = createTimeoutController(timeoutMs);

  try {
    const formData = new FormData();

    request.files.forEach((file) => {
      formData.append('files', file);
    });

    formData.append('complexity_threshold', String(request.complexityThreshold ?? getDefaultComplexityThreshold()));
    formData.append('force_system2', String(request.forceSystem2 ?? false));
    formData.append('enable_verification', String(request.enableVerification ?? false));

    const response = await fetch(`${MODAL_API_URL}/extract/batch`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      await handleResponseError(response);
    }

    const data = await response.json();

    // Validate response
    return validateBatchResponse(data);
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError('Request timed out', 0, 'timeout');
    }
    if (error instanceof ApiError || error instanceof ValidationError) {
      throw error;
    }
    throw new ApiError(
      error instanceof Error ? error.message : 'Unknown error',
      0,
      'network'
    );
  } finally {
    clearTimeout(timeoutId);
  }
};

export const healthCheck = async (): Promise<{ status: string; supabase_enabled?: boolean }> => {
  const response = await fetch(`${MODAL_API_URL}/health`);
  if (!response.ok) {
    throw new ApiError('Health check failed', response.status, 'server');
  }
  return response.json();
};
