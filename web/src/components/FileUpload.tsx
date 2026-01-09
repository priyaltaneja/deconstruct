import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, FileText, Loader2 } from 'lucide-react';
import { Button } from './ui/button';
import { useExtractionStore } from '../store/extractionStore';
import { getApiUrl } from '../config/environment';

interface FileWithPreview extends File {
  error?: string;
}

const MAX_FILE_SIZE = 50 * 1024 * 1024;
const ACCEPTED_TYPES = {
  'application/pdf': ['.pdf'],
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
};

export function FileUpload() {
  const [files, setFiles] = useState<FileWithPreview[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const createBatch = useExtractionStore((s) => s.createBatch);
  const addJob = useExtractionStore((s) => s.addJob);
  const updateJob = useExtractionStore((s) => s.updateJob);
  const setActiveBatch = useExtractionStore((s) => s.setActiveBatch);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setUploadError(null);
    const newFiles: FileWithPreview[] = acceptedFiles.map((file) => {
      if (file.size > MAX_FILE_SIZE) {
        return Object.assign(file, { error: 'File too large (max 50MB)' });
      }
      return file;
    });
    rejectedFiles.forEach((rejected) => {
      const file = rejected.file as FileWithPreview;
      file.error = rejected.errors[0]?.message || 'Invalid file';
      newFiles.push(file);
    });
    setFiles((prev) => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_FILE_SIZE,
  });

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const validFiles = files.filter((f) => !f.error);

  const handleUpload = async () => {
    if (validFiles.length === 0) return;

    setIsUploading(true);
    setUploadError(null);

    try {
      const batchName = validFiles.length === 1
        ? validFiles[0].name
        : `${validFiles.length} documents`;
      const batchId = createBatch(batchName);

      const jobIds: string[] = [];
      for (const file of validFiles) {
        const jobId = addJob({
          batchId,
          fileName: file.name,
          fileSize: file.size,
          status: 'queued',
        });
        jobIds.push(jobId);
      }

      setActiveBatch(batchId);

      const apiUrl = getApiUrl();

      for (let i = 0; i < validFiles.length; i++) {
        const file = validFiles[i];
        const jobId = jobIds[i];

        updateJob(jobId, { status: 'processing' });

        try {
          const formData = new FormData();
          formData.append('file', file);
          formData.append('document_id', file.name);

          const response = await fetch(`${apiUrl}/extract`, {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }

          const result = await response.json();

          updateJob(jobId, {
            status: 'completed',
            documentType: result.document_type,
            reasoningTier: result.reasoning_tier,
            modelUsed: result.model_used,
            extractedData: result,
            confidence: result.confidence_score,
            processingTimeMs: result.processing_time_ms,
            costUsd: result.cost_usd,
          });
        } catch (err) {
          updateJob(jobId, {
            status: 'failed',
            error: err instanceof Error ? err.message : 'Extraction failed',
          });
        }
      }

      setFiles([]);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="space-y-6">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-xl p-10 text-center cursor-pointer
          transition-all duration-200 ease-out
          ${isDragActive
            ? 'border-primary bg-primary/5 scale-[1.01]'
            : 'border-border hover:border-muted-foreground/50 hover:bg-muted/50'
          }
          ${isUploading ? 'pointer-events-none opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-3">
          <div className={`
            w-12 h-12 rounded-full flex items-center justify-center
            ${isDragActive ? 'bg-primary/10 text-primary' : 'bg-muted text-muted-foreground'}
            transition-colors
          `}>
            <Upload className="w-5 h-5" />
          </div>
          <div>
            <p className="font-medium">
              {isDragActive ? 'Drop files here' : 'Drop files or click to browse'}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              PDF, PNG, or JPEG up to 50MB
            </p>
          </div>
        </div>
      </div>

      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((file, index) => (
            <div
              key={`${file.name}-${index}`}
              className={`
                flex items-center gap-3 p-3 rounded-lg border bg-card
                ${file.error ? 'border-destructive/30 bg-destructive/5' : 'border-border'}
              `}
            >
              <div className={`
                w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0
                ${file.error ? 'bg-destructive/10 text-destructive' : 'bg-muted text-muted-foreground'}
              `}>
                <FileText className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{file.name}</p>
                <p className={`text-xs ${file.error ? 'text-destructive' : 'text-muted-foreground'}`}>
                  {file.error || formatFileSize(file.size)}
                </p>
              </div>
              <button
                onClick={() => removeFile(index)}
                className="p-1.5 rounded-md hover:bg-muted transition-colors"
              >
                <X className="w-4 h-4 text-muted-foreground" />
              </button>
            </div>
          ))}
        </div>
      )}

      {uploadError && (
        <p className="text-sm text-destructive bg-destructive/10 px-3 py-2 rounded-lg">
          {uploadError}
        </p>
      )}

      {validFiles.length > 0 && (
        <Button onClick={handleUpload} disabled={isUploading} className="w-full" size="lg">
          {isUploading ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Processing...
            </>
          ) : (
            `Extract ${validFiles.length} file${validFiles.length !== 1 ? 's' : ''}`
          )}
        </Button>
      )}
    </div>
  );
}
