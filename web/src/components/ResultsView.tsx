import { useExtractionStore, ExtractionJob } from '../store/extractionStore';
import { Button } from './ui/button';
import { ArrowLeft, Copy, Check, Loader2, FileText, CheckCircle, XCircle, Clock } from 'lucide-react';
import { useState } from 'react';

interface ResultsViewProps {
  batchId: string;
}

export function ResultsView({ batchId }: ResultsViewProps) {
  const jobs = useExtractionStore((s) => s.jobs);
  const batches = useExtractionStore((s) => s.batches);
  const setActiveBatch = useExtractionStore((s) => s.setActiveBatch);

  const batch = batches.get(batchId);
  const batchJobs = Array.from(jobs.values()).filter((j) => j.batchId === batchId);

  const [selectedJob, setSelectedJob] = useState<string | null>(batchJobs[0]?.id || null);
  const [copied, setCopied] = useState(false);

  if (!batch) {
    return <p className="text-muted-foreground">Batch not found</p>;
  }

  const completedJobs = batchJobs.filter((j) => j.status === 'completed').length;
  const failedJobs = batchJobs.filter((j) => j.status === 'failed').length;
  const selectedJobData = selectedJob ? jobs.get(selectedJob) : null;

  const copyToClipboard = async () => {
    if (selectedJobData?.extractedData) {
      await navigator.clipboard.writeText(
        JSON.stringify(selectedJobData.extractedData, null, 2)
      );
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const getStatusIcon = (status: ExtractionJob['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-emerald-600" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-destructive" />;
      case 'processing':
        return <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />;
      default:
        return <Clock className="w-4 h-4 text-muted-foreground" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" onClick={() => setActiveBatch(null)}>
            <ArrowLeft className="w-4 h-4 mr-1.5" />
            Back
          </Button>
          <div className="h-4 w-px bg-border" />
          <div className="text-sm text-muted-foreground">
            {completedJobs} of {batchJobs.length} complete
            {failedJobs > 0 && <span className="text-destructive ml-1">({failedJobs} failed)</span>}
          </div>
        </div>
      </div>

      {/* Main grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* File list */}
        <div className="border border-border rounded-xl bg-card overflow-hidden">
          <div className="px-4 py-3 border-b border-border bg-muted/30">
            <h3 className="text-sm font-medium">Documents</h3>
          </div>
          <div className="divide-y divide-border max-h-[500px] overflow-y-auto">
            {batchJobs.map((job) => (
              <button
                key={job.id}
                onClick={() => setSelectedJob(job.id)}
                className={`
                  w-full flex items-center gap-3 px-4 py-3 text-left transition-colors
                  ${selectedJob === job.id ? 'bg-muted' : 'hover:bg-muted/50'}
                `}
              >
                <div className="w-8 h-8 rounded-lg bg-muted flex items-center justify-center flex-shrink-0">
                  <FileText className="w-4 h-4 text-muted-foreground" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{job.fileName}</p>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    {getStatusIcon(job.status)}
                    <span className="text-xs text-muted-foreground capitalize">{job.status}</span>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Result detail */}
        <div className="lg:col-span-2 border border-border rounded-xl bg-card overflow-hidden">
          <div className="px-4 py-3 border-b border-border bg-muted/30 flex items-center justify-between">
            <h3 className="text-sm font-medium truncate">
              {selectedJobData?.fileName || 'Select a document'}
            </h3>
            {selectedJobData?.status === 'completed' && (
              <Button variant="ghost" size="sm" onClick={copyToClipboard}>
                {copied ? (
                  <>
                    <Check className="w-4 h-4 mr-1.5" />
                    Copied
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4 mr-1.5" />
                    Copy
                  </>
                )}
              </Button>
            )}
          </div>

          <div className="p-4">
            {selectedJobData ? (
              <>
                {selectedJobData.status === 'failed' && (
                  <div className="text-sm text-destructive bg-destructive/10 px-3 py-2 rounded-lg">
                    {selectedJobData.error || 'Extraction failed'}
                  </div>
                )}

                {selectedJobData.status === 'processing' && (
                  <div className="flex items-center justify-center py-12 text-muted-foreground">
                    <Loader2 className="w-5 h-5 animate-spin mr-2" />
                    Processing document...
                  </div>
                )}

                {selectedJobData.status === 'queued' && (
                  <div className="flex items-center justify-center py-12 text-muted-foreground">
                    <Clock className="w-5 h-5 mr-2" />
                    Waiting in queue...
                  </div>
                )}

                {selectedJobData.status === 'completed' && selectedJobData.extractedData && (
                  <pre className="text-sm bg-muted/50 p-4 rounded-lg overflow-auto max-h-[450px]">
                    {JSON.stringify(selectedJobData.extractedData, null, 2)}
                  </pre>
                )}
              </>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                <FileText className="w-8 h-8 mb-2 opacity-50" />
                <p className="text-sm">Select a document to view results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
