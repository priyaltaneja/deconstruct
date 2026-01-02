/**
 * Main App Component
 * Modern dark mode dashboard for monitoring GPU pipeline
 */

import React from 'react';
import { FileUpload } from './components/FileUpload';
import { JobMonitor } from './components/JobMonitor';
import { useExtractionStore } from './store/extractionStore';

function App() {
  const batches = useExtractionStore((state) => state.batches);
  const activeBatchId = useExtractionStore((state) => state.activeBatchId);
  const setActiveBatch = useExtractionStore((state) => state.setActiveBatch);

  const styles = {
    container: {
      minHeight: '100vh',
      backgroundColor: 'var(--bg-primary)',
    },
    header: {
      background: 'transparent',
      borderBottom: '1px solid var(--border-color)',
      padding: '20px 32px',
      position: 'sticky' as const,
      top: 0,
      zIndex: 10,
      backdropFilter: 'blur(10px)',
      backgroundColor: 'rgba(10, 14, 26, 0.8)',
    },
    headerContent: {
      maxWidth: '1200px',
      margin: '0 auto',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    },
    logo: {
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
    },
    logoIcon: {
      fontSize: '1.5em',
      color: 'var(--accent-blue)',
    },
    title: {
      margin: 0,
      fontSize: '1.5em',
      fontWeight: 600,
      color: 'var(--text-primary)',
      letterSpacing: '-0.3px',
    },
    batchSelector: {
      backgroundColor: 'rgba(17, 24, 39, 0.6)',
      padding: '14px 32px',
      borderBottom: '1px solid var(--border-color)',
      backdropFilter: 'blur(10px)',
    },
    batchSelectorInner: {
      maxWidth: '1200px',
      margin: '0 auto',
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
    },
    label: {
      color: 'var(--text-secondary)',
      fontSize: '0.9em',
      fontWeight: 500,
    },
    select: {
      padding: '10px 16px',
      borderRadius: '8px',
      border: '1px solid var(--border-color)',
      fontSize: '14px',
      backgroundColor: 'var(--bg-tertiary)',
      color: 'var(--text-primary)',
      minWidth: '300px',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      outline: 'none',
    },
    mainContent: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: activeBatchId ? '32px' : '24px 32px',
      display: 'grid',
      gridTemplateColumns: activeBatchId ? '420px 1fr' : '1fr',
      gap: '24px',
    },
    card: {
      backgroundColor: 'var(--bg-secondary)',
      borderRadius: '16px',
      padding: '32px',
      boxShadow: 'var(--shadow-lg)',
      border: '1px solid var(--border-color)',
      transition: 'all 0.3s ease',
    },
    cardTitle: {
      marginTop: 0,
      marginBottom: '24px',
      fontSize: '1.25em',
      fontWeight: 600,
      color: 'var(--text-primary)',
    },
    welcomeSection: {
      textAlign: 'center' as const,
      padding: '0 20px 16px',
      marginBottom: '0',
    },
    welcomeTitle: {
      fontSize: '1.5em',
      fontWeight: 500,
      marginBottom: '8px',
      color: 'var(--text-primary)',
      letterSpacing: '-0.3px',
    },
    welcomeText: {
      fontSize: '1em',
      color: 'var(--text-secondary)',
      maxWidth: '700px',
      margin: '0 auto',
      lineHeight: '1.6',
    },
    uploadSection: {
      maxWidth: '1000px',
      margin: '0 auto',
      width: '100%',
    },
  };

  return (
    <div style={styles.container}>
      {/* Minimal Header */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.logo}>
            <div style={styles.logoIcon}>⚡</div>
            <h1 style={styles.title}>Deconstruct</h1>
          </div>
        </div>
      </header>

      {/* Batch Selector */}
      {batches.size > 0 && (
        <div style={styles.batchSelector}>
          <div style={styles.batchSelectorInner}>
            <label style={styles.label}>Active Batch:</label>
            <select
              value={activeBatchId || ''}
              onChange={(e) => setActiveBatch(e.target.value || null)}
              style={styles.select}
              onMouseOver={(e) => {
                (e.target as HTMLSelectElement).style.borderColor = 'var(--accent-blue)';
              }}
              onMouseOut={(e) => {
                (e.target as HTMLSelectElement).style.borderColor = 'var(--border-color)';
              }}
            >
              <option value="">Select a batch...</option>
              {Array.from(batches.values())
                .sort(
                  (a, b) =>
                    new Date(b.createdAt).getTime() -
                    new Date(a.createdAt).getTime()
                )
                .map((batch) => (
                  <option key={batch.id} value={batch.id}>
                    {batch.name} ({batch.completedJobs}/{batch.totalJobs})
                  </option>
                ))}
            </select>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div style={styles.mainContent}>
        {!activeBatchId ? (
          <>
            {/* Welcome Section */}
            <div style={styles.welcomeSection}>
              <h2 style={styles.welcomeTitle}>Welcome to Deconstruct</h2>
              <p style={styles.welcomeText}>
                Upload your documents to start extracting structured data.
              </p>
            </div>

            {/* Upload Section */}
            <div style={styles.uploadSection}>
              <div style={{ ...styles.card, padding: '40px' }}>
                <FileUpload />
              </div>
            </div>
          </>
        ) : (
          <>
            {/* Upload Section */}
            <div>
              <div style={styles.card}>
                <h2 style={styles.cardTitle}>Upload Documents</h2>
                <FileUpload />
              </div>

              {/* Quick Stats */}
              <div style={{ ...styles.card, marginTop: '24px' }}>
                <h3 style={styles.cardTitle}>System Status</h3>
                <div style={{ fontSize: '0.95em' }}>
                  <StatRow
                    label="Total Batches"
                    value={batches.size}
                  />
                  <StatRow
                    label="Active Jobs"
                    value={useExtractionStore.getState().getActiveJobs().length}
                  />
                </div>
              </div>
            </div>

            {/* Job Monitor */}
            <JobMonitor />
          </>
        )}
      </div>
    </div>
  );
}

const StatRow: React.FC<{ label: string; value: number }> = ({ label, value }) => (
  <div
    style={{
      display: 'flex',
      justifyContent: 'space-between',
      padding: '12px 16px',
      marginBottom: '8px',
      backgroundColor: 'var(--bg-tertiary)',
      borderRadius: '8px',
      border: '1px solid var(--border-color)',
    }}
  >
    <span style={{ color: 'var(--text-secondary)' }}>{label}:</span>
    <strong style={{ color: 'var(--accent-blue)', fontSize: '1.1em' }}>
      {value}
    </strong>
  </div>
);


export default App;
