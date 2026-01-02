/**
 * Drag-and-Drop File Upload Component
 * Modern dark mode with blue accents
 */

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { validateFile, formatFileSize } from '../schemas/validation';
import { useExtractionStore } from '../store/extractionStore';
import { extractBatch } from '../services/api';

interface FileWithPreview extends File {
  preview?: string;
  validationError?: string;
}

export const FileUpload: React.FC = () => {
  const [files, setFiles] = useState<FileWithPreview[]>([]);
  const [uploading, setUploading] = useState(false);

  const createBatch = useExtractionStore((state) => state.createBatch);
  const addJob = useExtractionStore((state) => state.addJob);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const validatedFiles = acceptedFiles.map((file) => {
      const validation = validateFile(file);
      const fileWithPreview: FileWithPreview = Object.assign(file, {
        validationError: validation.valid ? undefined : validation.error,
      });
      return fileWithPreview;
    });

    setFiles((prev) => [...prev, ...validatedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
    },
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    const validFiles = files.filter((f) => !f.validationError);
    if (validFiles.length === 0) {
      alert('No valid files to upload');
      return;
    }

    setUploading(true);

    try {
      // Create batch
      const batchId = createBatch(`Batch ${new Date().toLocaleString()}`);

      // Create job entries
      const jobIds: string[] = [];
      validFiles.forEach((file) => {
        const jobId = addJob({
          batchId,
          fileName: file.name,
          fileSize: file.size,
          status: 'queued',
        });
        jobIds.push(jobId);
      });

      // Update status to processing
      const updateJob = useExtractionStore.getState().updateJob;
      jobIds.forEach((id) => updateJob(id, { status: 'processing' }));

      // Send to Modal API
      console.log('Sending batch to Modal API...');
      const results = await extractBatch({
        files: validFiles,
        complexityThreshold: 0.8,
        forceSystem2: false,
        enableVerification: false,
      });

      // Update jobs with results
      results.forEach((result: any, index: number) => {
        const jobId = jobIds[index];
        updateJob(jobId, {
          status: 'completed',
          documentType: result.document_type,
          reasoningTier: result.reasoning_tier,
          modelUsed: result.model_used,
          processingTimeMs: result.processing_time_ms,
          costUsd: result.cost_usd,
          confidence: result.confidence_score,
          verificationStatus: result.verification_status,
          complexityMarkers: result.complexity_markers,
          extractedData: result,
        });
      });

      setFiles([]);
      alert(`Successfully processed ${validFiles.length} file(s)!`);
    } catch (error) {
      console.error('Upload failed:', error);
      alert(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);

      // Mark jobs as failed
      const updateJob = useExtractionStore.getState().updateJob;
      const jobs = useExtractionStore.getState().jobs;
      jobs.forEach((job) => {
        if (job.status === 'processing') {
          updateJob(job.id, {
            status: 'failed',
            error: error instanceof Error ? error.message : 'Unknown error',
          });
        }
      });
    } finally {
      setUploading(false);
    }
  };

  const styles = {
    dropzone: {
      border: '2px dashed var(--border-color)',
      borderRadius: '16px',
      padding: '60px 40px',
      textAlign: 'center' as const,
      cursor: 'pointer',
      backgroundColor: isDragActive ? 'var(--bg-tertiary)' : 'transparent',
      transition: 'all 0.3s ease',
      position: 'relative' as const,
      overflow: 'hidden' as const,
    },
    dropzoneActive: {
      borderColor: 'var(--accent-blue)',
      backgroundColor: 'rgba(59, 130, 246, 0.05)',
      boxShadow: '0 0 0 4px rgba(59, 130, 246, 0.1)',
    },
    uploadIcon: {
      fontSize: '4em',
      marginBottom: '24px',
      color: 'var(--accent-blue)',
      opacity: 0.9,
    },
    uploadText: {
      color: 'var(--text-primary)',
      fontSize: '1.2em',
      marginBottom: '12px',
      fontWeight: 500,
    },
    uploadSubtext: {
      fontSize: '0.95em',
      color: 'var(--text-muted)',
      lineHeight: '1.5',
    },
    fileList: {
      marginTop: '32px',
    },
    fileListHeader: {
      color: 'var(--text-primary)',
      fontSize: '1.1em',
      marginBottom: '16px',
      fontWeight: 600,
    },
    fileListContainer: {
      maxHeight: '320px',
      overflowY: 'auto' as const,
    },
    fileItem: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '12px 16px',
      border: '1px solid var(--border-color)',
      borderRadius: '8px',
      marginBottom: '8px',
      backgroundColor: 'var(--bg-tertiary)',
      transition: 'all 0.2s ease',
    },
    fileItemError: {
      borderColor: 'var(--status-error)',
      backgroundColor: 'rgba(239, 68, 68, 0.1)',
    },
    fileName: {
      fontWeight: 500,
      color: 'var(--text-primary)',
      marginBottom: '4px',
    },
    fileSize: {
      fontSize: '0.85em',
      color: 'var(--text-secondary)',
    },
    fileError: {
      fontSize: '0.85em',
      color: 'var(--status-error)',
      marginTop: '4px',
    },
    removeButton: {
      padding: '6px 12px',
      cursor: 'pointer',
      backgroundColor: 'var(--status-error)',
      color: 'white',
      border: 'none',
      borderRadius: '6px',
      fontSize: '0.875em',
      fontWeight: 500,
      transition: 'all 0.2s ease',
    },
    uploadButton: {
      marginTop: '24px',
      padding: '16px 32px',
      backgroundColor: uploading ? 'var(--border-color)' : 'var(--accent-blue)',
      color: 'white',
      border: 'none',
      borderRadius: '12px',
      cursor: uploading ? 'not-allowed' : 'pointer',
      fontSize: '1em',
      fontWeight: 600,
      width: '100%',
      transition: 'all 0.2s ease',
      boxShadow: uploading ? 'none' : 'var(--shadow-md)',
    },
  };

  return (
    <div>
      <div
        {...getRootProps()}
        style={{
          ...styles.dropzone,
          ...(isDragActive ? styles.dropzoneActive : {}),
        }}
        onMouseOver={(e) => {
          if (!isDragActive) {
            e.currentTarget.style.borderColor = 'var(--accent-blue-light)';
          }
        }}
        onMouseOut={(e) => {
          if (!isDragActive) {
            e.currentTarget.style.borderColor = 'var(--border-color)';
          }
        }}
      >
        <input {...getInputProps()} />
        <div style={styles.uploadIcon}>📄</div>
        {isDragActive ? (
          <p style={styles.uploadText}>Drop files here</p>
        ) : (
          <div>
            <p style={styles.uploadText}>Drag & drop files here</p>
            <p style={styles.uploadSubtext}>
              or click to select files • Max 50MB per file
              <br />
              <span style={{ fontSize: '0.9em', color: 'var(--text-muted)' }}>
                PDF, PNG, JPEG supported
              </span>
            </p>
          </div>
        )}
      </div>

      {files.length > 0 && (
        <div style={styles.fileList}>
          <h3 style={styles.fileListHeader}>
            Files ({files.length})
          </h3>
          <div style={styles.fileListContainer}>
            {files.map((file, index) => (
              <div
                key={index}
                style={{
                  ...styles.fileItem,
                  ...(file.validationError ? styles.fileItemError : {}),
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.transform = 'translateX(4px)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.transform = 'translateX(0)';
                }}
              >
                <div style={{ flex: 1 }}>
                  <div style={styles.fileName}>{file.name}</div>
                  <div style={styles.fileSize}>
                    {formatFileSize(file.size)}
                  </div>
                  {file.validationError && (
                    <div style={styles.fileError}>
                      {file.validationError}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => removeFile(index)}
                  style={styles.removeButton}
                  onMouseOver={(e) => {
                    e.currentTarget.style.backgroundColor = '#dc2626';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.backgroundColor = 'var(--status-error)';
                  }}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>

          <button
            onClick={handleUpload}
            disabled={uploading || files.every((f) => f.validationError)}
            style={styles.uploadButton}
            onMouseOver={(e) => {
              if (!uploading && !files.every((f) => f.validationError)) {
                e.currentTarget.style.backgroundColor = 'var(--accent-blue-hover)';
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = 'var(--shadow-lg)';
              }
            }}
            onMouseOut={(e) => {
              if (!uploading) {
                e.currentTarget.style.backgroundColor = 'var(--accent-blue)';
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = 'var(--shadow-md)';
              }
            }}
          >
            {uploading ? 'Uploading...' : `Upload ${files.length} File${files.length > 1 ? 's' : ''}`}
          </button>
        </div>
      )}
    </div>
  );
};
