import { serve } from 'inngest/next';
import { inngest } from '@/lib/inngest/client';
import { runTracesEvalFunction } from '@/lib/inngest/eval/run-traces-eval';
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 800;

export const { GET, POST, PUT } = serve({client: inngest,
    functions: [
      // Evals
      runTracesEvalFunction,
    ],
    streaming: 'force',
  });
  