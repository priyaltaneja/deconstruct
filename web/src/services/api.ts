/**
 * API Service for Modal Backend
 * Handles document extraction requests
 */

// Use local API that calls Modal via subprocess
const MODAL_API_URL = 'http://localhost:8000';

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

export const extractDocument = async (request: ExtractionRequest) => {
  const formData = new FormData();
  formData.append('file', request.file);
  formData.append('complexity_threshold', String(request.complexityThreshold ?? 0.8));
  formData.append('force_system2', String(request.forceSystem2 ?? false));

  const response = await fetch(`${MODAL_API_URL}/api/extract`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Extraction failed: ${error}`);
  }

  return response.json();
};

export const extractBatch = async (request: BatchExtractionRequest) => {
  const formData = new FormData();

  request.files.forEach((file) => {
    formData.append('files', file);
  });

  formData.append('complexity_threshold', String(request.complexityThreshold ?? 0.8));
  formData.append('force_system2', String(request.forceSystem2 ?? false));
  formData.append('enable_verification', String(request.enableVerification ?? false));

  const response = await fetch(`${MODAL_API_URL}/api/extract/batch`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Batch extraction failed: ${error}`);
  }

  return response.json();
};

export const healthCheck = async () => {
  const response = await fetch(`${MODAL_API_URL}/api/health`);
  return response.json();
};
