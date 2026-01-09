import { useExtractionStore, ExtractionJob } from '../store/extractionStore';
import { Button } from './ui/button';
import { ArrowLeft, Copy, Check, Loader2, FileText, CheckCircle, XCircle, Clock, Cpu, Eye, Zap, ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';

interface ResultsViewProps {
  batchId: string;
}

interface ReasoningStep {
  step: number;
  action: string;
  status: string;
  duration_ms?: number;
  details?: Record<string, any>;
}

export function ResultsView({ batchId }: ResultsViewProps) {
  const jobs = useExtractionStore((s) => s.jobs);
  const batches = useExtractionStore((s) => s.batches);
  const setActiveBatch = useExtractionStore((s) => s.setActiveBatch);

  const batch = batches.get(batchId);
  const batchJobs = Array.from(jobs.values()).filter((j) => j.batchId === batchId);

  const [selectedJob, setSelectedJob] = useState<string | null>(batchJobs[0]?.id || null);
  const [copied, setCopied] = useState(false);
  const [showReasoning, setShowReasoning] = useState(true);
  const [showJson, setShowJson] = useState(false);

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

  const extractedData = selectedJobData?.extractedData || {};
  const reasoningSteps: ReasoningStep[] = extractedData.reasoning_steps || [];

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
          <div className="divide-y divide-border max-h-[600px] overflow-y-auto">
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
        <div className="lg:col-span-2 space-y-4">
          {selectedJobData ? (
            <>
              {/* Reasoning Panel */}
              {selectedJobData.status === 'completed' && reasoningSteps.length > 0 && (
                <div className="border border-border rounded-xl bg-card overflow-hidden">
                  <button
                    onClick={() => setShowReasoning(!showReasoning)}
                    className="w-full px-4 py-3 border-b border-border bg-muted/30 flex items-center justify-between hover:bg-muted/50 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-blue-500" />
                      <h3 className="text-sm font-medium">Processing Pipeline</h3>
                    </div>
                    {showReasoning ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                  </button>

                  {showReasoning && (
                    <div className="p-4 space-y-3">
                      {reasoningSteps.map((step, idx) => (
                        <div key={idx} className="flex gap-3">
                          <div className="flex flex-col items-center">
                            <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-medium ${
                              step.status === 'complete' ? 'bg-emerald-100 text-emerald-700' : 'bg-blue-100 text-blue-700'
                            }`}>
                              {step.step}
                            </div>
                            {idx < reasoningSteps.length - 1 && (
                              <div className="w-px h-full bg-border mt-1" />
                            )}
                          </div>
                          <div className="flex-1 pb-3">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium">{step.action}</span>
                              {step.duration_ms !== undefined && (
                                <span className="text-xs text-muted-foreground">
                                  {step.duration_ms}ms
                                </span>
                              )}
                            </div>
                            {step.details && (
                              <div className="mt-1.5 text-xs text-muted-foreground space-y-0.5">
                                {Object.entries(step.details).map(([key, value]) => (
                                  <div key={key} className="flex gap-2">
                                    <span className="text-muted-foreground/70">{key.replace(/_/g, ' ')}:</span>
                                    <span className={
                                      key === 'selected_tier'
                                        ? value === 'system2'
                                          ? 'text-amber-600 font-medium'
                                          : 'text-emerald-600 font-medium'
                                        : ''
                                    }>
                                      {typeof value === 'number'
                                        ? key.includes('score') || key.includes('confidence')
                                          ? `${(value * 100).toFixed(1)}%`
                                          : value.toLocaleString()
                                        : String(value)
                                      }
                                    </span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}

                      {/* Summary stats */}
                      <div className="flex gap-4 pt-3 border-t border-border">
                        <div className="flex items-center gap-2">
                          {extractedData.reasoning_tier === 'system2' ? (
                            <Eye className="w-4 h-4 text-amber-500" />
                          ) : (
                            <Zap className="w-4 h-4 text-emerald-500" />
                          )}
                          <span className="text-xs">
                            {extractedData.reasoning_tier === 'system2' ? 'Vision LLM' : 'Text LLM'}
                          </span>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {extractedData.processing_time_ms?.toFixed(0)}ms
                        </div>
                        <div className="text-xs text-muted-foreground">
                          ${extractedData.cost_usd?.toFixed(4)}
                        </div>
                        <div className={`text-xs font-medium ${
                          (extractedData.confidence_score || 0) >= 0.8
                            ? 'text-emerald-600'
                            : 'text-amber-600'
                        }`}>
                          {((extractedData.confidence_score || 0) * 100).toFixed(0)}% confidence
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* JSON Output */}
              <div className="border border-border rounded-xl bg-card overflow-hidden">
                <div className="px-4 py-3 border-b border-border bg-muted/30 flex items-center justify-between">
                  <button
                    onClick={() => setShowJson(!showJson)}
                    className="flex items-center gap-2 hover:opacity-70 transition-opacity"
                  >
                    <h3 className="text-sm font-medium">Extracted Data</h3>
                    {showJson ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                  </button>
                  {selectedJobData.status === 'completed' && (
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

                  {selectedJobData.status === 'completed' && selectedJobData.extractedData && showJson && (
                    <pre className="text-xs bg-muted/50 p-4 rounded-lg overflow-auto max-h-[400px]">
                      {JSON.stringify(
                        // Filter out reasoning_steps and page_analysis for cleaner output
                        Object.fromEntries(
                          Object.entries(selectedJobData.extractedData).filter(
                            ([key]) => !['reasoning_steps', 'page_analysis'].includes(key)
                          )
                        ),
                        null,
                        2
                      )}
                    </pre>
                  )}

                  {selectedJobData.status === 'completed' && !showJson && (
                    <p className="text-sm text-muted-foreground">Click to expand JSON output</p>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="border border-border rounded-xl bg-card p-8">
              <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                <FileText className="w-8 h-8 mb-2 opacity-50" />
                <p className="text-sm">Select a document to view results</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
