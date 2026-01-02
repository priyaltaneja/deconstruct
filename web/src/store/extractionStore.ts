/**
 * Zustand Store for Extraction Job Management
 * Persists state to IndexedDB using localForage
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import localforage from 'localforage';

// Configure localForage
localforage.config({
  name: 'deconstruct',
  storeName: 'extraction_jobs',
});

export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed' | 'verifying';

export type ReasoningTier = 'system1' | 'system2';

export interface ComplexityMarkers {
  has_nested_tables: boolean;
  has_multi_column_layout: boolean;
  has_handwriting: boolean;
  has_low_quality_scan: boolean;
  has_ambiguous_language: boolean;
  has_complex_formulas: boolean;
  language_is_mixed: boolean;
  page_count: number;
  estimated_entities: number;
  complexity_score: number;
}

export interface ExtractionJob {
  id: string;
  batchId: string;
  fileName: string;
  fileSize: number;
  status: JobStatus;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;

  // Extraction metadata
  documentType?: 'legal' | 'financial' | 'technical' | 'general';
  complexityMarkers?: ComplexityMarkers;
  reasoningTier?: ReasoningTier;
  modelUsed?: string;

  // Results
  extractedData?: any;
  confidence?: number;
  verificationStatus?: 'passed' | 'failed' | 'needs_review';

  // Performance metrics
  processingTimeMs?: number;
  costUsd?: number;

  // Error handling
  error?: string;
  retryCount: number;
}

export interface Batch {
  id: string;
  name: string;
  createdAt: string;
  totalJobs: number;
  completedJobs: number;
  failedJobs: number;
  totalCostUsd: number;
  totalProcessingTimeMs: number;
}

interface ExtractionStore {
  // State
  jobs: Map<string, ExtractionJob>;
  batches: Map<string, Batch>;
  activeBatchId: string | null;

  // Actions
  createBatch: (name: string) => string;
  addJob: (job: Omit<ExtractionJob, 'id' | 'createdAt' | 'retryCount'>) => string;
  updateJob: (jobId: string, updates: Partial<ExtractionJob>) => void;
  deleteJob: (jobId: string) => void;
  deleteBatch: (batchId: string) => void;
  setActiveBatch: (batchId: string | null) => void;

  // Selectors
  getJobsByBatch: (batchId: string) => ExtractionJob[];
  getActiveJobs: () => ExtractionJob[];
  getBatchStats: (batchId: string) => Batch | undefined;
}

export const useExtractionStore = create<ExtractionStore>()(
  persist(
    (set, get) => ({
      jobs: new Map(),
      batches: new Map(),
      activeBatchId: null,

      createBatch: (name: string) => {
        const batchId = `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const batch: Batch = {
          id: batchId,
          name,
          createdAt: new Date().toISOString(),
          totalJobs: 0,
          completedJobs: 0,
          failedJobs: 0,
          totalCostUsd: 0,
          totalProcessingTimeMs: 0,
        };

        set((state) => ({
          batches: new Map(state.batches).set(batchId, batch),
          activeBatchId: batchId,
        }));

        return batchId;
      },

      addJob: (jobData) => {
        const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const job: ExtractionJob = {
          ...jobData,
          id: jobId,
          createdAt: new Date().toISOString(),
          retryCount: 0,
        };

        set((state) => {
          const newJobs = new Map(state.jobs).set(jobId, job);

          // Update batch stats
          const batch = state.batches.get(job.batchId);
          if (batch) {
            const updatedBatch = { ...batch, totalJobs: batch.totalJobs + 1 };
            const newBatches = new Map(state.batches).set(job.batchId, updatedBatch);
            return { jobs: newJobs, batches: newBatches };
          }

          return { jobs: newJobs };
        });

        return jobId;
      },

      updateJob: (jobId, updates) => {
        set((state) => {
          const job = state.jobs.get(jobId);
          if (!job) return state;

          const updatedJob = { ...job, ...updates };
          const newJobs = new Map(state.jobs).set(jobId, updatedJob);

          // Update batch stats
          const batch = state.batches.get(job.batchId);
          if (batch) {
            let completedDelta = 0;
            let failedDelta = 0;
            let costDelta = 0;
            let timeDelta = 0;

            // Check if status changed to completed or failed
            if (updates.status === 'completed' && job.status !== 'completed') {
              completedDelta = 1;
              costDelta = updates.costUsd || 0;
              timeDelta = updates.processingTimeMs || 0;
            } else if (updates.status === 'failed' && job.status !== 'failed') {
              failedDelta = 1;
            }

            const updatedBatch: Batch = {
              ...batch,
              completedJobs: batch.completedJobs + completedDelta,
              failedJobs: batch.failedJobs + failedDelta,
              totalCostUsd: batch.totalCostUsd + costDelta,
              totalProcessingTimeMs: batch.totalProcessingTimeMs + timeDelta,
            };

            const newBatches = new Map(state.batches).set(job.batchId, updatedBatch);
            return { jobs: newJobs, batches: newBatches };
          }

          return { jobs: newJobs };
        });
      },

      deleteJob: (jobId) => {
        set((state) => {
          const newJobs = new Map(state.jobs);
          newJobs.delete(jobId);
          return { jobs: newJobs };
        });
      },

      deleteBatch: (batchId) => {
        set((state) => {
          // Delete all jobs in the batch
          const newJobs = new Map(state.jobs);
          state.jobs.forEach((job, id) => {
            if (job.batchId === batchId) {
              newJobs.delete(id);
            }
          });

          // Delete the batch
          const newBatches = new Map(state.batches);
          newBatches.delete(batchId);

          return {
            jobs: newJobs,
            batches: newBatches,
            activeBatchId: state.activeBatchId === batchId ? null : state.activeBatchId,
          };
        });
      },

      setActiveBatch: (batchId) => {
        set({ activeBatchId: batchId });
      },

      getJobsByBatch: (batchId) => {
        const jobs = get().jobs;
        return Array.from(jobs.values()).filter((job) => job.batchId === batchId);
      },

      getActiveJobs: () => {
        const jobs = get().jobs;
        return Array.from(jobs.values()).filter(
          (job) => job.status === 'queued' || job.status === 'processing'
        );
      },

      getBatchStats: (batchId) => {
        return get().batches.get(batchId);
      },
    }),
    {
      name: 'extraction-storage',
      storage: createJSONStorage(() => localforage),
      // Serialize Map to Array for storage
      partialize: (state) => ({
        jobs: Array.from(state.jobs.entries()),
        batches: Array.from(state.batches.entries()),
        activeBatchId: state.activeBatchId,
      }),
      // Deserialize Array back to Map
      merge: (persistedState: any, currentState) => ({
        ...currentState,
        jobs: new Map(persistedState.jobs || []),
        batches: new Map(persistedState.batches || []),
        activeBatchId: persistedState.activeBatchId || null,
      }),
    }
  )
);
