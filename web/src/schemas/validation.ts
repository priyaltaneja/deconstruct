/**
 * Zod Validation Schemas for Deconstruct
 * Frontend validation that mirrors backend Pydantic schemas
 */

import { z } from 'zod';

// ============ FILE UPLOAD VALIDATION ============
export const fileUploadSchema = z.object({
  file: z
    .instanceof(File)
    .refine((file) => file.size <= 50 * 1024 * 1024, {
      message: 'File size must be less than 50MB',
    })
    .refine(
      (file) => ['application/pdf', 'image/png', 'image/jpeg'].includes(file.type),
      {
        message: 'Only PDF, PNG, and JPEG files are allowed',
      }
    ),
  metadata: z
    .object({
      documentType: z
        .enum(['legal', 'financial', 'technical', 'general'])
        .optional(),
      priority: z.enum(['low', 'normal', 'high']).default('normal'),
      tags: z.array(z.string()).optional(),
    })
    .optional(),
});

export type FileUpload = z.infer<typeof fileUploadSchema>;

// ============ BATCH REQUEST VALIDATION ============
export const batchRequestSchema = z.object({
  batchName: z
    .string()
    .min(1, 'Batch name is required')
    .max(100, 'Batch name too long'),
  files: z
    .array(fileUploadSchema)
    .min(1, 'At least one file is required')
    .max(100, 'Maximum 100 files per batch'),
  settings: z.object({
    forceSystem2: z.boolean().default(false),
    complexityThreshold: z.number().min(0).max(1).default(0.8),
    enableVerification: z.boolean().default(true),
    maxRetries: z.number().min(0).max(5).default(2),
  }),
});

export type BatchRequest = z.infer<typeof batchRequestSchema>;

// ============ COMPLEXITY MARKERS VALIDATION ============
export const complexityMarkersSchema = z.object({
  has_nested_tables: z.boolean(),
  has_multi_column_layout: z.boolean(),
  has_handwriting: z.boolean(),
  has_low_quality_scan: z.boolean(),
  has_ambiguous_language: z.boolean(),
  has_complex_formulas: z.boolean(),
  language_is_mixed: z.boolean(),
  page_count: z.number().int().positive(),
  estimated_entities: z.number().int().nonnegative(),
  complexity_score: z.number().min(0).max(1),
});

export type ComplexityMarkers = z.infer<typeof complexityMarkersSchema>;

// ============ EXTRACTION RESULT VALIDATION ============
export const extractionResultSchema = z.object({
  document_id: z.string(),
  document_type: z.enum(['legal', 'financial', 'technical', 'general']),
  complexity_markers: complexityMarkersSchema,
  reasoning_tier: z.enum(['system1', 'system2']),
  model_used: z.string(),
  processing_time_ms: z.number().positive(),
  cost_usd: z.number().nonnegative(),
  confidence_score: z.number().min(0).max(1),
  verification_status: z.enum(['passed', 'failed', 'needs_review']),
  verification_notes: z.array(z.string()).optional(),

  // Content (one of these will be populated based on document_type)
  legal_content: z.any().optional(),
  financial_content: z.any().optional(),
  technical_content: z.any().optional(),
  raw_text: z.string().optional(),
  extracted_entities: z.array(z.any()).optional(),
});

export type ExtractionResult = z.infer<typeof extractionResultSchema>;

// ============ API RESPONSE VALIDATION ============
export const apiResponseSchema = z.object({
  success: z.boolean(),
  data: z.any().optional(),
  error: z
    .object({
      code: z.string(),
      message: z.string(),
      details: z.any().optional(),
    })
    .optional(),
  meta: z
    .object({
      timestamp: z.string(),
      request_id: z.string(),
      processing_time_ms: z.number().optional(),
    })
    .optional(),
});

export type ApiResponse = z.infer<typeof apiResponseSchema>;

// ============ VALIDATION HELPERS ============

/**
 * Validate file before upload
 */
export function validateFile(file: File): { valid: boolean; error?: string } {
  const result = fileUploadSchema.safeParse({ file });
  if (!result.success) {
    return {
      valid: false,
      error: result.error.errors[0].message,
    };
  }
  return { valid: true };
}

/**
 * Validate batch request before submission
 */
export function validateBatchRequest(
  request: unknown
): { valid: boolean; data?: BatchRequest; error?: string } {
  const result = batchRequestSchema.safeParse(request);
  if (!result.success) {
    return {
      valid: false,
      error: result.error.errors.map((e) => e.message).join(', '),
    };
  }
  return { valid: true, data: result.data };
}

/**
 * Parse and validate extraction result from API
 */
export function parseExtractionResult(
  data: unknown
): { valid: boolean; data?: ExtractionResult; error?: string } {
  const result = extractionResultSchema.safeParse(data);
  if (!result.success) {
    return {
      valid: false,
      error: 'Invalid extraction result format',
    };
  }
  return { valid: true, data: result.data };
}

/**
 * File size formatter
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Cost formatter
 */
export function formatCost(usd: number): string {
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

/**
 * Duration formatter
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`;
}
