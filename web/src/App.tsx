import { FileUpload } from './components/FileUpload';
import { ResultsView } from './components/ResultsView';
import { useExtractionStore } from './store/extractionStore';

function App() {
  const batches = useExtractionStore((state) => state.batches);
  const activeBatchId = useExtractionStore((state) => state.activeBatchId);
  const setActiveBatch = useExtractionStore((state) => state.setActiveBatch);

  if (activeBatchId) {
    return (
      <div className="min-h-screen bg-background">
        <header className="border-b border-border bg-card">
          <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
            <button
              onClick={() => setActiveBatch(null)}
              className="font-semibold tracking-tight hover:opacity-70 transition-opacity"
            >
              Deconstruct
            </button>
            {batches.size > 0 && (
              <select
                value={activeBatchId || ''}
                onChange={(e) => setActiveBatch(e.target.value || null)}
                className="bg-background border border-border rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-1"
              >
                <option value="">New extraction</option>
                {Array.from(batches.values())
                  .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
                  .map((batch) => (
                    <option key={batch.id} value={batch.id}>
                      {batch.name} ({batch.completedJobs}/{batch.totalJobs})
                    </option>
                  ))}
              </select>
            )}
          </div>
        </header>
        <main className="max-w-5xl mx-auto px-6 py-10">
          <ResultsView batchId={activeBatchId} />
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <div className="w-full max-w-xl space-y-8">
        <div className="text-center space-y-3">
          <h1 className="text-4xl font-semibold tracking-tight">Deconstruct</h1>
          <p className="text-muted-foreground">
            Extract structured data from documents. Vision LLMs run on Modal servers—your data never leaves your infrastructure.
          </p>
        </div>
        <FileUpload />
      </div>
    </div>
  );
}

export default App;
