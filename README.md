# Langfuse Prompt Evaluation UI

**Test prompt changes against production data before deploying them.**

This application provides a simple UI to draft prompt variations, run them against your most recent production traces from Langfuse, and compare results side-by-side. Works with OpenAI and Google Gemini models.

## What This Does

- **Draft prompts with spell-check**: Edit prompt templates with built-in English spell checking to catch typos before evaluation
- **Run evals on real data**: Test prompt variations against actual production traces to see how they would have performed
- **Side-by-side comparison**: View original vs. new responses for each trace to evaluate improvements
- **Fine-tune OpenAI models**: Convert your best traces into training data and kick off fine-tuning jobs directly from the UI
- **Multi-provider support**: Run evaluations with both OpenAI and Google Gemini models

## How It Works

This tool expects you to log prompt metadata to your Langfuse observations in production. When you make LLM calls using `observeOpenAI` or similar wrappers, you need to include specific fields in the **observation metadata**:

1. **`promptVariables`** - The input variables passed to your prompt (required for all evals)
2. **`tools`** - Function calling definitions if your prompt uses function calling (required for tool-based prompts)
3. **`responseFormat`** - JSON schema or structured output requirements (required for structured output prompts)

By logging this metadata in your production traces, the eval system can:
- Extract the original input variables from any trace
- Replay the same call with your modified prompt template
- Use the same tools and response format constraints
- Compare the new output against the original production output

### Example: Logging Metadata with OpenAI

Here's how to pass prompt variables and metadata through a Langfuse-wrapped OpenAI call:

```typescript
import { observeOpenAI } from "@langfuse/openai";
import OpenAI from "openai";

const openai = observeOpenAI(new OpenAI());

// Your prompt variables
const promptVariables = {
  userQuery: "What's the weather like?",
  context: "User is located in San Francisco"
};

// Define your tools (if using function calling)
const tools = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get current weather",
      parameters: {
        type: "object",
        properties: {
          location: { type: "string" }
        },
        required: ["location"]
      }
    }
  }
];

// Make the API call with metadata
const response = await openai.chat.completions.create(
  {
    model: "gpt-4",
    messages: [
      {
        role: "system",
        content: "You are a helpful weather assistant."
      },
      {
        role: "user",
        content: promptVariables.userQuery
      }
    ],
    tools: tools,
    // Optional: for structured outputs (OpenAI JSON schema)
    response_format: { type: "json_object" }
  },
  {
    // Langfuse observeOpenAI options
    parent: trace, // optional: parent trace
    generationName: "weather-assistant",
    metadata: {
      // REQUIRED: Log prompt variables for eval replay
      promptVariables: promptVariables,
      // REQUIRED (if using tools): Log tools for eval replay
      tools: tools,
      // REQUIRED (if using structured output): Log response format schema
      responseFormat: { type: "json_object" }
    }
  }
);
```

**Important**: The eval system looks for these fields in the **observation metadata**:

- `metadata.promptVariables` - Required for all evals (the input variables)
- `metadata.tools` - Required if your prompt uses function calling
- `metadata.responseFormat` - Required if your prompt uses structured outputs

The eval runner will automatically extract these from your production traces and replay them with your modified prompt template.

**Metadata Lookup Priority**: The system searches for `promptVariables` in this order:
1. Trace-level metadata (`trace.metadata.promptVariables`)
2. Trace input (`trace.input.promptVariables`)
3. Observation metadata (`observation.metadata.promptVariables`)

Tools and schemas are always extracted from observation metadata where they were used.

## Requirements

### Required Services

- **Node.js 18+** and pnpm (or npm/yarn)
- **[Langfuse](https://langfuse.com)** account and project
- **API keys** for OpenAI and/or Google Gemini

### Optional Services (for deployment)

- **[Inngest](https://inngest.com)** - For async eval execution (free tier available)
- **[Vercel Blob Storage](https://vercel.com/docs/storage/vercel-blob)** - For storing eval results and training data

> **Note**: You can run this entirely locally without Inngest or Vercel Blob if you prefer. Simply run `pnpm dev` and access the UI at `http://localhost:3000/evals`. However, you'll need these services if deploying to production or wanting background job processing.

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd next-lang-evals
pnpm install
```

### 2. Configure Environment Variables

Copy the example environment file:

```bash
cp env.example .env.local
```

Edit `.env.local` with your credentials:

```bash
# ============================================
# REQUIRED: Langfuse Configuration
# ============================================
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

# Find your project ID in your Langfuse project URL
# Example: https://us.cloud.langfuse.com/project/abc123def456/prompts
#          Your project ID is: abc123def456
NEXT_PUBLIC_LANGFUSE_PROJECT_ID=your-project-id-here
NEXT_PUBLIC_LANGFUSE_BASE_URL=https://us.cloud.langfuse.com

# ============================================
# REQUIRED: At least one LLM provider API key
# ============================================
# OpenAI (required for fine-tuning features)
OPENAI_API_KEY=sk-...

# Google Gemini (optional, for Gemini model support)
GEMINI_API_KEY=...

# ============================================
# OPTIONAL: Required only for deployment
# ============================================
# Vercel Blob Storage (for storing eval results)
BLOB_READ_WRITE_TOKEN=vercel_blob_...

# Inngest (for background job processing)
INNGEST_EVENT_KEY=...
INNGEST_SIGNING_KEY=...
```

### 3. Find Your Langfuse Project ID

1. Log into [Langfuse](https://cloud.langfuse.com)
2. Navigate to any page in your project (Prompts, Traces, etc.)
3. Look at the URL in your browser address bar
4. Copy the string after `/project/`

Example:
```
https://us.cloud.langfuse.com/project/cme8s6dl501fbad07hhojh4r6/prompts
                                     ^^^^^^^^^^^^^^^^^^^^^^^^
                                     This is your project ID
```

### 4. Run Locally

```bash
pnpm dev
```

The application will be available at:
- **Evals UI**: [http://localhost:3000/evals](http://localhost:3000/evals)
- **Fine-tuning**: [http://localhost:3000/evals/finetune](http://localhost:3000/evals/finetune)

### 5. Deploy to Vercel (Optional)

This application is designed to deploy seamlessly to Vercel:

```bash
# Install Vercel CLI
pnpm add -g vercel

# Deploy
vercel
```

Configure the same environment variables in your Vercel project settings. You'll also need to:

1. Set up [Vercel Blob Storage](https://vercel.com/docs/storage/vercel-blob) and add the `BLOB_READ_WRITE_TOKEN`
2. Configure [Inngest](https://www.inngest.com/docs/deploy/vercel) with your deployment URL and add `INNGEST_EVENT_KEY` and `INNGEST_SIGNING_KEY`

## Usage Guide

### Running Evaluations

1. Navigate to `/evals` to browse your Langfuse prompts
2. Click on a prompt to view its details and usage statistics
3. Go to the **Draft & Eval** tab
4. Edit the prompt template (spell-check will highlight any issues)
5. Select recent traces to test against
6. Choose your model provider and model
7. Click **Run Eval** to execute
8. Review side-by-side comparisons of original vs. new responses

### Fine-tuning OpenAI Models

1. Navigate to `/evals/finetune`
2. Select a prompt to extract training examples from its traces
3. Review the generated training data
4. Adjust filters and parameters as needed
5. Save training data to blob storage
6. Start an OpenAI fine-tuning job directly from the UI

## Technical Architecture

This application is built with:

- **Frontend**: Next.js 15 App Router with React Server Components
- **Styling**: Tailwind CSS with shadcn/ui components
- **Background Jobs**: Inngest for async eval execution (optional)
- **Observability**: Langfuse for prompt management and trace tracking
- **Storage**: Vercel Blob for eval results and training data (optional)
- **LLM Providers**: OpenAI and Google Gemini

### Project Structure

```
app/
  ├── evals/              # Main evals UI
  │   ├── page.tsx        # Prompt browser
  │   ├── compare/        # Eval comparison view
  │   ├── finetune/       # Fine-tuning UI
  │   └── components/     # Eval UI components
  ├── api/
  │   ├── evals/          # Eval execution API routes
  │   ├── langfuse/       # Langfuse proxy endpoints
  │   └── inngest/        # Inngest webhook handler
lib/
  ├── inngest/
  │   └── eval/           # Background eval execution logic
  ├── ai/                 # LLM provider wrappers
  │   ├── langfuse-wrapped-openai.ts
  │   ├── langfuse-wrapped-gemini.ts
  │   └── schema-registry.ts
  └── observability/      # Langfuse client helpers
```

## Troubleshooting

### "No traces found for this prompt"

Make sure you're logging the required metadata to your Langfuse observations. When using `observeOpenAI` or other Langfuse wrappers, pass these fields in the `metadata` option:

```typescript
const response = await openai.chat.completions.create(
  { /* your request */ },
  {
    generationName: "my-generation",
    metadata: {
      promptVariables: { /* your input variables */ },
      tools: [ /* your tools array if using function calling */ ],
      responseFormat: { /* your schema if using structured outputs */ }
    }
  }
);
```

The eval system looks for:
- `metadata.promptVariables` - The variables passed to your prompt template (required)
- `metadata.tools` - Function calling definitions (required if using tools)
- `metadata.responseFormat` - Structured output schema (required if using structured outputs)

### Evals timing out or not running

If running locally without Inngest, evals execute synchronously and may take longer for prompts with many traces. Consider:
- Limiting the number of traces selected for evaluation
- Using Inngest for background processing if running many evals
- Increasing your API rate limits with your LLM provider

### Fine-tuning job fails to start

Ensure:
- Your `OPENAI_API_KEY` has access to fine-tuning endpoints
- You have sufficient credits in your OpenAI account
- Your training data meets OpenAI's formatting requirements (automatically validated by the UI)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT
