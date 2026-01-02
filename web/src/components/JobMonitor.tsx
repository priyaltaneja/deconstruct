/**
 * Job Monitoring Dashboard
 * Real-time tracking with modern dark mode design
 */

import React from 'react';
import { useExtractionStore } from '../store/extractionStore';
import { formatCost, formatDuration } from '../schemas/validation';

export const JobMonitor: React.FC = () => {
  const { jobs, batches, activeBatchId, getBatchStats, getJobsByBatch } =
    useExtractionStore();

  const activeBatch = activeBatchId ? getBatchStats(activeBatchId) : null;
  const activeJobs = activeBatchId ? getJobsByBatch(activeBatchId) : [];

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'completed':
        return 'var(--status-success)';
      case 'processing':
        return 'var(--status-info)';
      case 'failed':
        return 'var(--status-error)';
      case 'verifying':
        return 'var(--status-warning)';
      default:
        return 'var(--text-muted)';
    }
  };

  const getReasoningTierBadge = (tier?: string) => {
    if (!tier) return null;
    const isSys2 = tier === 'system2';
    return (
      <span
        style={{
          padding: '4px 10px',
          borderRadius: '6px',
          fontSize: '0.75em',
          fontWeight: 600,
          backgroundColor: isSys2 ? 'rgba(139, 92, 246, 0.2)' : 'rgba(59, 130, 246, 0.2)',
          color: isSys2 ? '#a78bfa' : 'var(--accent-blue-light)',
          border: `1px solid ${isSys2 ? 'rgba(167, 139, 250, 0.3)' : 'rgba(59, 130, 246, 0.3)'}`,
        }}
      >
        {isSys2 ? 'System 2' : 'System 1'}
      </span>
    );
  };

  if (!activeBatch) {
    return (
      <div style={{
        padding: '60px 20px',
        textAlign: 'center',
        color: 'var(--text-muted)'
      }}>
        <p>No active batch. Upload files to get started.</p>
      </div>
    );
  }

  const progressPercentage =
    activeBatch.totalJobs > 0
      ? ((activeBatch.completedJobs + activeBatch.failedJobs) /
          activeBatch.totalJobs) *
        100
      : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Batch Overview Card */}
      <div style={{
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: '16px',
        padding: '32px',
        boxShadow: 'var(--shadow-lg)',
        border: '1px solid var(--border-color)',
      }}>
        <h2 style={{
          marginTop: 0,
          marginBottom: '20px',
          color: 'var(--text-primary)',
          fontSize: '1.5em',
          fontWeight: 600,
        }}>
          {activeBatch.name}
        </h2>

        {/* Progress Bar */}
        <div style={{
          width: '100%',
          height: '10px',
          backgroundColor: 'var(--bg-tertiary)',
          borderRadius: '6px',
          overflow: 'hidden',
          marginBottom: '20px',
          border: '1px solid var(--border-color)',
        }}>
          <div
            style={{
              width: `${progressPercentage}%`,
              height: '100%',
              background: 'linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-blue-light) 100%)',
              transition: 'width 0.3s ease',
              boxShadow: '0 0 10px rgba(59, 130, 246, 0.5)',
            }}
          />
        </div>

        {/* Stats Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '12px',
        }}>
          <StatCard label="Total Jobs" value={activeBatch.totalJobs} color="var(--text-primary)" />
          <StatCard label="Completed" value={activeBatch.completedJobs} color="var(--status-success)" />
          <StatCard label="Failed" value={activeBatch.failedJobs} color="var(--status-error)" />
          <StatCard label="Total Cost" value={formatCost(activeBatch.totalCostUsd)} color="var(--accent-blue)" />
          <StatCard label="Total Time" value={formatDuration(activeBatch.totalProcessingTimeMs)} color="var(--accent-blue-light)" />
        </div>
      </div>

      {/* Jobs List Card */}
      <div style={{
        backgroundColor: 'var(--bg-secondary)',
        borderRadius: '16px',
        padding: '32px',
        boxShadow: 'var(--shadow-lg)',
        border: '1px solid var(--border-color)',
      }}>
        <h3 style={{
          marginTop: 0,
          marginBottom: '16px',
          color: 'var(--text-primary)',
          fontSize: '1.25em',
          fontWeight: 600,
        }}>
          Jobs ({activeJobs.length})
        </h3>

        <div style={{
          maxHeight: '600px',
          overflowY: 'auto',
        }}>
          {activeJobs.length === 0 ? (
            <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0' }}>
              No jobs yet
            </p>
          ) : (
            activeJobs.map((job) => (
              <JobCard key={job.id} job={job} getStatusColor={getStatusColor} getReasoningTierBadge={getReasoningTierBadge} />
            ))
          )}
        </div>
      </div>
    </div>
  );
};

const StatCard: React.FC<{ label: string; value: string | number; color: string }> = ({
  label,
  value,
  color,
}) => (
  <div style={{
    backgroundColor: 'var(--bg-tertiary)',
    padding: '20px',
    borderRadius: '12px',
    border: '1px solid var(--border-color)',
    transition: 'all 0.2s ease',
  }}
  onMouseOver={(e) => {
    e.currentTarget.style.borderColor = 'var(--accent-blue)';
    e.currentTarget.style.transform = 'translateY(-2px)';
  }}
  onMouseOut={(e) => {
    e.currentTarget.style.borderColor = 'var(--border-color)';
    e.currentTarget.style.transform = 'translateY(0)';
  }}>
    <div style={{
      fontSize: '0.8em',
      color: 'var(--text-muted)',
      marginBottom: '8px',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      fontWeight: 500,
    }}>
      {label}
    </div>
    <div style={{
      fontSize: '1.75em',
      fontWeight: 700,
      color: color,
    }}>
      {value}
    </div>
  </div>
);

const JobCard: React.FC<{
  job: any;
  getStatusColor: (status: string) => string;
  getReasoningTierBadge: (tier?: string) => JSX.Element | null;
}> = ({ job, getStatusColor, getReasoningTierBadge }) => {
  const [showJson, setShowJson] = React.useState(false);

  const downloadJson = () => {
    if (!job.extractedData) return;

    const blob = new Blob([JSON.stringify(job.extractedData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${job.fileName.replace('.pdf', '')}_extracted.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
  <div
    style={{
      border: '1px solid var(--border-color)',
      borderRadius: '12px',
      padding: '20px',
      marginBottom: '12px',
      backgroundColor: 'var(--bg-tertiary)',
      transition: 'all 0.2s ease',
    }}
    onMouseOver={(e) => {
      e.currentTarget.style.borderColor = 'var(--accent-blue)';
      e.currentTarget.style.transform = 'translateX(4px)';
    }}
    onMouseOut={(e) => {
      e.currentTarget.style.borderColor = 'var(--border-color)';
      e.currentTarget.style.transform = 'translateX(0)';
    }}
  >
    {/* Header */}
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      marginBottom: '12px',
    }}>
      <div style={{ flex: 1 }}>
        <div style={{
          fontWeight: 600,
          marginBottom: '6px',
          color: 'var(--text-primary)',
          fontSize: '1.05em',
        }}>
          {job.fileName}
        </div>
        <div style={{
          fontSize: '0.85em',
          color: 'var(--text-secondary)',
          display: 'flex',
          gap: '12px',
          flexWrap: 'wrap',
        }}>
          {job.documentType && (
            <span>
              Type: <span style={{ color: 'var(--accent-blue-light)' }}>{job.documentType}</span>
            </span>
          )}
          {job.complexityMarkers && (
            <span>
              Complexity: <span style={{ color: 'var(--accent-blue-light)' }}>
                {(job.complexityMarkers.complexity_score * 100).toFixed(0)}%
              </span>
            </span>
          )}
        </div>
      </div>

      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        {getReasoningTierBadge(job.reasoningTier)}
        <span
          style={{
            padding: '6px 14px',
            borderRadius: '6px',
            fontSize: '0.8em',
            fontWeight: 600,
            backgroundColor: `${getStatusColor(job.status)}20`,
            color: getStatusColor(job.status),
            border: `1px solid ${getStatusColor(job.status)}40`,
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
          }}
        >
          {job.status}
        </span>
      </div>
    </div>

    {/* Details */}
    {job.status === 'completed' && (
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
        gap: '12px',
        marginTop: '12px',
        paddingTop: '12px',
        borderTop: '1px solid var(--border-color)',
        fontSize: '0.85em',
      }}>
        <DetailItem label="Model" value={job.modelUsed} />
        <DetailItem
          label="Time"
          value={job.processingTimeMs ? formatDuration(job.processingTimeMs) : 'N/A'}
        />
        <DetailItem
          label="Cost"
          value={job.costUsd !== undefined ? formatCost(job.costUsd) : 'N/A'}
          highlight
        />
        <DetailItem
          label="Confidence"
          value={job.confidence ? `${(job.confidence * 100).toFixed(0)}%` : 'N/A'}
        />
        {job.verificationStatus && (
          <DetailItem
            label="Verification"
            value={job.verificationStatus}
            color={
              job.verificationStatus === 'passed'
                ? 'var(--status-success)'
                : job.verificationStatus === 'failed'
                ? 'var(--status-error)'
                : 'var(--status-warning)'
            }
          />
        )}
      </div>
    )}

    {/* Error */}
    {job.status === 'failed' && job.error && (
      <div style={{
        marginTop: '12px',
        padding: '12px',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderRadius: '6px',
        fontSize: '0.85em',
        color: 'var(--status-error)',
        border: '1px solid rgba(239, 68, 68, 0.3)',
      }}>
        <strong>Error:</strong> {job.error}
      </div>
    )}

    {/* View JSON Button */}
    {job.status === 'completed' && job.extractedData && (
      <div style={{
        marginTop: '12px',
        display: 'flex',
        gap: '8px',
      }}>
        <button
          onClick={() => setShowJson(!showJson)}
          style={{
            flex: 1,
            padding: '10px 16px',
            backgroundColor: 'var(--bg-tertiary)',
            color: 'var(--accent-blue)',
            border: '1px solid var(--accent-blue)',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '0.875em',
            fontWeight: 600,
            transition: 'all 0.2s ease',
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
          }}
        >
          {showJson ? 'Hide JSON' : 'View JSON'}
        </button>
        <button
          onClick={downloadJson}
          style={{
            flex: 1,
            padding: '10px 16px',
            backgroundColor: 'var(--accent-blue)',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '0.875em',
            fontWeight: 600,
            transition: 'all 0.2s ease',
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--accent-blue-hover)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--accent-blue)';
          }}
        >
          Download JSON
        </button>
      </div>
    )}

    {/* JSON Viewer */}
    {showJson && job.extractedData && (
      <div style={{
        marginTop: '12px',
        padding: '16px',
        backgroundColor: '#0a0e1a',
        borderRadius: '8px',
        border: '1px solid var(--border-color)',
        maxHeight: '400px',
        overflowY: 'auto',
      }}>
        <pre style={{
          margin: 0,
          fontSize: '0.8em',
          color: '#e5e7eb',
          lineHeight: '1.6',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}>
          {JSON.stringify(job.extractedData, null, 2)}
        </pre>
      </div>
    )}
  </div>
);
};

const DetailItem: React.FC<{
  label: string;
  value: string;
  color?: string;
  highlight?: boolean;
}> = ({ label, value, color, highlight }) => (
  <div>
    <span style={{ color: 'var(--text-muted)', fontSize: '0.9em' }}>{label}: </span>
    <strong
      style={{
        color: color || (highlight ? 'var(--accent-blue)' : 'var(--text-primary)'),
        fontSize: highlight ? '1.1em' : '1em',
      }}
    >
      {value}
    </strong>
  </div>
);
