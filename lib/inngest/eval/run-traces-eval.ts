import { langfuseWrappedGemini } from '@/lib/ai/langfuse-wrapped-gemini';
import { langfuseWrappedOpenAI } from '@/lib/ai/langfuse-wrapped-openai';
import { inngest } from '@/lib/inngest/client';
import { openaiToolsToGeminiTools } from '@/lib/ai/openai-to-gemini-tools';
import {
  getSchemaFromObservation,
  getToolsFromObservation,
  OpenAITool,
} from '@/lib/ai/schema-registry';
import { zodToGeminiSchema } from '@/lib/ai/zod-to-gemini-schema';
import { getLangfuse } from '@/lib/observability/langfuse';
import { storeDataInBlob } from '@/lib/utils/blob-storage';
import { FunctionCall } from '@google/genai';
import { v4 as uuidv4 } from 'uuid';
import { z } from 'zod';

// Supported model providers
type ModelProvider = 'openai' | 'gemini';

// Trace-observation pair for direct lookup
interface TraceObservationPair {
  traceId: string;
  observationId: string;
  // Original output can be passed directly to avoid re-fetching
  output?: any;
}

// Input for a single eval run
interface EvalRunInput {
  // The draft prompt template with {{variable}} placeholders
  draftPrompt: string;
  // Trace IDs to load and run against (legacy, use traceObservationPairs when available)
  traceIds: string[];
  // Trace-observation pairs for direct lookup (preferred over traceIds)
  traceObservationPairs?: TraceObservationPair[];
  // Model configuration
  model: string;
  provider: ModelProvider;
  // Name for this eval run
  evalName: string;
  // Original prompt name/version for reference
  originalPromptName: string;
  originalPromptVersion: number;
  // Max concurrent runs
  concurrency?: number;
  // Pre-generated Langfuse trace ID (allows deeplink before run completes)
  langfuseTraceId?: string;
  // OpenAI reasoning parameters (for GPT-5 and reasoning models)
  reasoningEffort?: 'minimal' | 'low' | 'medium' | 'high';
  verbosity?: 'low' | 'medium' | 'high';
  // Schema key for structured output validation (matches keys in schema-registry.ts)
  schemaKey?: string;
  // Tool group key for tool-calling evals (matches keys in evalToolsRegistry)
  toolGroupKey?: string;
}

// Result for a single trace evaluation
interface TraceEvalResult {
  traceId: string;
  success: boolean;
  input?: Record<string, any>;
  output?: string;
  // Original production output from the trace being evaluated
  originalProductionOutput?: string;
  // Parsed structured output (when schemaKey is provided)
  parsed?: Record<string, any> | null;
  error?: string;
  latencyMs?: number;
  variableSource?: string;
  tokenUsage?: {
    input?: number;
    output?: number;
    total?: number;
  };
  // Format detected: 'json' (array), 'text-markers' ([SYSTEM]/[USER]), or 'plain'
  promptFormat?: 'json' | 'text-markers' | 'plain';
  // Schema key used for validation (if any)
  schemaKey?: string;
  // Tool group key used for tool-calling (if any)
  toolGroupKey?: string;
  // Tool calls made by the model (if tools were available)
  toolCalls?: Array<{
    id: string;
    name: string;
    arguments: Record<string, any>;
  }>;
}

// Overall eval run result
interface EvalRunResult {
  evalId: string;
  evalName: string;
  totalTraces: number;
  successCount: number;
  failureCount: number;
  results: TraceEvalResult[];
  startedAt: string;
  completedAt: string;
  durationMs: number;
  // URL to full results JSON in blob storage
  resultsUrl?: string;
}

/**
 * Fill template variables in a string.
 * Supports {{variable}} and {variable} syntax.
 * Also handles variable names with underscores vs hyphens.
 */
function fillVariables(
  template: string,
  variables: Record<string, any>,
): string {
  let result = template;

  for (const [key, value] of Object.entries(variables)) {
    const stringValue =
      value !== null && value !== undefined ? String(value) : '';

    // Handle {{variable}} syntax (exact match)
    result = result.replace(new RegExp(`\\{\\{${key}\\}\\}`, 'g'), stringValue);
    // Handle {variable} syntax (exact match)
    result = result.replace(new RegExp(`\\{${key}\\}`, 'g'), stringValue);

    // Also try with underscores/hyphens normalized for flexibility
    // If key has underscores, also try with hyphens
    const hyphenKey = key.replace(/_/g, '-');
    if (hyphenKey !== key) {
      result = result.replace(
        new RegExp(`\\{\\{${hyphenKey}\\}\\}`, 'g'),
        stringValue,
      );
      result = result.replace(
        new RegExp(`\\{${hyphenKey}\\}`, 'g'),
        stringValue,
      );
    }

    // If key has hyphens, also try with underscores
    const underscoreKey = key.replace(/-/g, '_');
    if (underscoreKey !== key) {
      result = result.replace(
        new RegExp(`\\{\\{${underscoreKey}\\}\\}`, 'g'),
        stringValue,
      );
      result = result.replace(
        new RegExp(`\\{${underscoreKey}\\}`, 'g'),
        stringValue,
      );
    }
  }

  // Replace any remaining unmatched {{variable}} patterns with empty strings
  // This ensures Langfuse shows filled messages, not placeholders
  result = result.replace(/\{\{[\w-]+\}\}/g, '');
  result = result.replace(/\{[\w-]+\}/g, '');

  return result;
}

// Chat message type
interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/**
 * Parse text-marker format prompts like:
 * [SYSTEM]
 * content here
 *
 * [USER]
 * content here
 */
function parseTextMarkerFormat(text: string): ChatMessage[] | null {
  // Check if it contains role markers
  const hasMarkers =
    text.includes('[SYSTEM]') ||
    text.includes('[USER]') ||
    text.includes('[ASSISTANT]');

  if (!hasMarkers) return null;

  const messages: ChatMessage[] = [];

  // Split by role markers, keeping the delimiter
  const parts = text.split(/(\[SYSTEM\]|\[USER\]|\[ASSISTANT\])/i);

  let currentRole: 'system' | 'user' | 'assistant' | null = null;

  for (const part of parts) {
    const trimmed = part.trim();
    if (!trimmed) continue;

    // Check if this part is a role marker
    const upperPart = trimmed.toUpperCase();
    if (upperPart === '[SYSTEM]') {
      currentRole = 'system';
    } else if (upperPart === '[USER]') {
      currentRole = 'user';
    } else if (upperPart === '[ASSISTANT]') {
      currentRole = 'assistant';
    } else if (currentRole) {
      // This is content for the current role
      messages.push({
        role: currentRole,
        content: trimmed,
      });
    }
  }

  return messages.length > 0 ? messages : null;
}

/**
 * Parse and fill a prompt template.
 * Handles multiple formats:
 * 1. JSON array format: [{"role": "system", "content": "..."}]
 * 2. Text marker format: [SYSTEM]\ncontent\n[USER]\ncontent
 * 3. Plain text
 *
 * Returns either an array of messages (for chat formats) or a single string.
 */
function parseAndFillPrompt(
  draftPrompt: string,
  variables: Record<string, any>,
):
  | { messages: ChatMessage[]; format: 'json' | 'text-markers' }
  | { text: string; format: 'plain' } {
  // Try to parse as JSON array (chat format)
  if (draftPrompt.trim().startsWith('[')) {
    try {
      const parsed = JSON.parse(draftPrompt);
      if (
        Array.isArray(parsed) &&
        parsed.length > 0 &&
        parsed[0].role &&
        typeof parsed[0].content === 'string'
      ) {
        // Fill variables in each message's content
        const filledMessages: ChatMessage[] = parsed.map((msg) => ({
          role: msg.role,
          content: fillVariables(msg.content, variables),
        }));
        return { messages: filledMessages, format: 'json' };
      }
    } catch {
      // Not valid JSON, continue to other formats
    }
  }

  // Try to parse as text-marker format
  const textMarkerMessages = parseTextMarkerFormat(draftPrompt);
  if (textMarkerMessages) {
    // Fill variables in each message's content
    const filledMessages: ChatMessage[] = textMarkerMessages.map((msg) => ({
      role: msg.role,
      content: fillVariables(msg.content, variables),
    }));
    return { messages: filledMessages, format: 'text-markers' };
  }

  // Plain text - fill variables and return as text
  return { text: fillVariables(draftPrompt, variables), format: 'plain' };
}

/**
 * Extract output from an observation - handles both string and object outputs
 */
function extractObservationOutput(obs: any): string | null {
  if (obs.output === null || obs.output === undefined) {
    return null;
  }
  if (typeof obs.output === 'string') {
    return obs.output;
  }
  if (typeof obs.output === 'object') {
    // For structured outputs, stringify the object
    return JSON.stringify(obs.output);
  }
  return null;
}

/**
 * Fetch a trace from Langfuse and extract input variables from metadata.promptVariables
 * Also returns the original output from the same generation where variables were found,
 * plus any schema and tools found in observation metadata.
 *
 * @param traceId - The Langfuse trace ID
 * @param originalPromptName - The prompt name to match (e.g., "workflow/intake-acceptance-criteria")
 *
 * Expected metadata structure (from generation):
 * {
 *   "promptVariables": {
 *     "work_orders_context": "...",
 *     // ... other variables
 *   },
 *   "schema": {  // Optional - for structured output
 *     "type": "object",
 *     "properties": { ... }
 *   },
 *   "tools": [  // Optional - for tool calling
 *     {
 *       "type": "function",
 *       "function": { "name": "...", "parameters": { ... } }
 *     }
 *   ]
 * }
 */
async function fetchTraceWithVariables(
  traceId: string,
  originalPromptName?: string,
): Promise<{
  variables: Record<string, any>;
  originalOutput: string | null;
  trace: any;
  source: string;
  schema?: z.ZodSchema<any>;
  tools?: OpenAITool[];
  observation?: any;
} | null> {
  const secretKey = process.env.LANGFUSE_SECRET_KEY;
  const publicKey = process.env.LANGFUSE_PUBLIC_KEY;
  const baseUrl =
    process.env.LANGFUSE_BASE_URL || 'https://us.cloud.langfuse.com';

  if (!secretKey || !publicKey) return null;

  const auth = Buffer.from(`${publicKey}:${secretKey}`).toString('base64');

  try {
    // Fetch trace first to check trace-level metadata
    const traceResponse = await fetch(
      `${baseUrl}/api/public/traces/${traceId}`,
      {
        headers: {
          Authorization: `Basic ${auth}`,
          'Content-Type': 'application/json',
        },
      },
    );

    if (!traceResponse.ok) {
      console.error(
        `[EvalRunner] Failed to fetch trace ${traceId}:`,
        traceResponse.status,
      );
      return null;
    }

    const trace = await traceResponse.json();

    // Always fetch observations to get the generation output
    const obsResponse = await fetch(
      `${baseUrl}/api/public/observations?traceId=${traceId}&type=GENERATION&limit=10`,
      {
        headers: {
          Authorization: `Basic ${auth}`,
          'Content-Type': 'application/json',
        },
      },
    );

    let observations: any[] = [];
    if (obsResponse.ok) {
      const obsData = await obsResponse.json();
      observations = obsData.data || [];
      console.log(
        `[EvalRunner] Found ${observations.length} observations for trace ${traceId}`,
      );
      console.log(
        `[EvalRunner] Looking for originalPromptName: "${originalPromptName}"`,
      );
      observations.forEach((obs, i) => {
        console.log(`[EvalRunner] Observation ${i}:`, {
          id: obs.id,
          name: obs.name,
          type: obs.type,
          model: obs.model,
          startTime: obs.startTime,
          // Log all metadata keys and promptName specifically
          metadataKeys: obs.metadata
            ? Object.keys(obs.metadata)
            : 'no metadata',
          promptName: obs.metadata?.promptName,
          // Log output type and first 100 chars
          outputType: typeof obs.output,
          outputPreview:
            typeof obs.output === 'string'
              ? obs.output.substring(0, 100)
              : typeof obs.output === 'object'
                ? JSON.stringify(obs.output).substring(0, 100)
                : 'null',
        });
      });
    } else {
      console.warn(
        `[EvalRunner] Failed to fetch observations for trace ${traceId}:`,
        obsResponse.status,
      );
    }

    // Helper to find the observation matching the prompt name
    // Matches by observation name containing the prompt name pattern
    const findMatchingObservation = () => {
      console.log(
        `[EvalRunner] findMatchingObservation called with originalPromptName: "${originalPromptName}"`,
      );
      console.log(
        `[EvalRunner] Available observations: ${observations.length}`,
      );

      if (!originalPromptName || observations.length === 0) {
        console.log(
          `[EvalRunner] No prompt name or no observations, returning first: ${observations[0]?.name || 'none'}`,
        );
        return observations[0] || null;
      }

      // Normalize prompt name for matching
      // e.g., "workflow/intake-acceptance-criteria" -> "intake-acceptance-criteria" -> "intakeacceptancecriteria"
      const promptBaseName =
        originalPromptName.split('/').pop() || originalPromptName;
      const promptNormalized = promptBaseName.toLowerCase().replace(/-/g, '');
      console.log(
        `[EvalRunner] Prompt base name: "${promptBaseName}", normalized: "${promptNormalized}"`,
      );

      // Try exact match first
      let match = observations.find((obs) => obs.name === originalPromptName);
      if (match) {
        console.log(
          `[EvalRunner] Found exact match observation: ${match.name}, id: ${match.id}`,
        );
        return match;
      }

      // Try case-insensitive match with normalization
      // This handles camelCase (generateMaintenanceIntakeAcceptanceCriteria) vs kebab-case (intake-acceptance-criteria)
      match = observations.find((obs) => {
        if (!obs.name) return false;
        const obsNormalized = obs.name.toLowerCase().replace(/-/g, '');
        return obsNormalized.includes(promptNormalized);
      });
      if (match) {
        console.log(
          `[EvalRunner] Found normalized match observation: ${match.name}, id: ${match.id}`,
        );
        return match;
      }

      // Try matching by function-name metadata
      match = observations.find((obs) => {
        const funcName = obs.metadata?.['function-name'];
        if (!funcName) return false;
        const funcNormalized = funcName.toLowerCase().replace(/-/g, '');
        return (
          funcNormalized.includes(promptNormalized) ||
          promptNormalized.includes(funcNormalized)
        );
      });
      if (match) {
        console.log(
          `[EvalRunner] Found function-name match observation: ${match.name}, id: ${match.id}`,
        );
        return match;
      }

      // Try matching by prompt metadata
      match = observations.find(
        (obs) =>
          obs.metadata?.promptName === originalPromptName ||
          obs.metadata?.promptName?.includes(promptBaseName),
      );
      if (match) {
        console.log(
          `[EvalRunner] Found metadata match observation: ${match.name}, id: ${match.id}`,
        );
        return match;
      }

      console.warn(
        `[EvalRunner] No matching observation for prompt "${originalPromptName}"`,
      );
      console.warn(
        `[EvalRunner] Observation names available: ${observations.map((o) => o.name).join(', ')}`,
      );
      console.warn(
        `[EvalRunner] Using first observation: ${observations[0]?.name}`,
      );
      return observations[0] || null;
    };

    // Check trace metadata for promptVariables (first priority - trace-level metadata)
    // Handle both nested (metadata.promptVariables) and direct (metadata is promptVariables) cases
    if (trace.metadata) {
      let variables: Record<string, any> | null = null;

      // Case 1: promptVariables is nested in metadata
      if (
        trace.metadata.promptVariables &&
        typeof trace.metadata.promptVariables === 'object' &&
        !Array.isArray(trace.metadata.promptVariables)
      ) {
        variables = trace.metadata.promptVariables;
      }
      // Case 2: metadata itself contains variable-like keys (promptVariables might be the metadata object)
      else if (
        typeof trace.metadata === 'object' &&
        !Array.isArray(trace.metadata) &&
        Object.keys(trace.metadata).length > 0
      ) {
        // Check if metadata has variable-like structure (all string values or objects)
        const hasVariableStructure = Object.values(trace.metadata).every(
          (v) => typeof v === 'string' || (typeof v === 'object' && v !== null),
        );
        if (hasVariableStructure) {
          variables = trace.metadata;
        }
      }

      if (variables && Object.keys(variables).length > 0) {
        console.log(
          `[EvalRunner] Found promptVariables in trace metadata for ${traceId}:`,
          Object.keys(variables),
        );
        // Get the original output from the matching observation
        const matchingObs = findMatchingObservation();
        const originalOutput = matchingObs
          ? extractObservationOutput(matchingObs)
          : null;
        console.log(
          `[EvalRunner] Original output from observation ${matchingObs?.name || 'none'}:`,
          originalOutput ? `${originalOutput.substring(0, 100)}...` : 'null',
        );
        
        // Extract schema and tools from observation metadata if available
        const schema = matchingObs ? getSchemaFromObservation(matchingObs) : undefined;
        const tools = matchingObs ? getToolsFromObservation(matchingObs) : undefined;
        
        if (schema) {
          console.log(`[EvalRunner] Found schema in observation metadata`);
        }
        if (tools) {
          console.log(`[EvalRunner] Found ${tools.length} tools in observation metadata`);
        }
        
        return { 
          variables, 
          originalOutput, 
          trace, 
          source: 'trace:metadata',
          schema,
          tools,
          observation: matchingObs
        };
      }
    }

    // Check trace input for promptVariables
    if (
      trace.input?.promptVariables &&
      typeof trace.input.promptVariables === 'object'
    ) {
      const variables = trace.input.promptVariables;
      if (Object.keys(variables).length > 0) {
        console.log(
          `[EvalRunner] Found promptVariables in trace input for ${traceId}:`,
          Object.keys(variables),
        );
        // Get the original output from the matching observation
        const matchingObs = findMatchingObservation();
        const originalOutput = matchingObs
          ? extractObservationOutput(matchingObs)
          : null;
        
        // Extract schema and tools from observation metadata if available
        const schema = matchingObs ? getSchemaFromObservation(matchingObs) : undefined;
        const tools = matchingObs ? getToolsFromObservation(matchingObs) : undefined;
        
        return { 
          variables, 
          originalOutput, 
          trace, 
          source: 'trace:input',
          schema,
          tools,
          observation: matchingObs
        };
      }
    }

    // Look for promptVariables in observation metadata (same observation has the output)
    for (const obs of observations) {
      if (
        obs.metadata?.promptVariables &&
        typeof obs.metadata.promptVariables === 'object'
      ) {
        const variables = obs.metadata.promptVariables;
        if (Object.keys(variables).length > 0) {
          console.log(
            `[EvalRunner] Found promptVariables in observation ${obs.id} for trace ${traceId}:`,
            Object.keys(variables),
          );
          // Get the original output from THIS SAME observation
          const originalOutput = extractObservationOutput(obs);
          console.log(
            `[EvalRunner] Original output from observation ${obs.id}:`,
            originalOutput ? `${originalOutput.substring(0, 100)}...` : 'null',
          );
          
          // Extract schema and tools from this observation's metadata
          const schema = getSchemaFromObservation(obs);
          const tools = getToolsFromObservation(obs);
          
          return {
            variables,
            originalOutput,
            trace: { traceId, observation: obs },
            source: `observation:${obs.id}`,
            schema,
            tools,
            observation: obs
          };
        }
      }
    }

    // Debug: log what we found in trace metadata
    console.warn(`[EvalRunner] No promptVariables found for trace ${traceId}`);
    if (trace.metadata) {
      console.warn(
        `[EvalRunner] Trace metadata keys:`,
        Object.keys(trace.metadata),
      );
      console.warn(
        `[EvalRunner] Trace metadata.promptVariables exists:`,
        !!trace.metadata.promptVariables,
      );
      console.warn(
        `[EvalRunner] Trace metadata.promptVariables type:`,
        typeof trace.metadata.promptVariables,
      );
      if (trace.metadata.promptVariables) {
        console.warn(
          `[EvalRunner] Trace metadata.promptVariables keys:`,
          typeof trace.metadata.promptVariables === 'object'
            ? Object.keys(trace.metadata.promptVariables)
            : 'not an object',
        );
      }
    } else {
      console.warn(`[EvalRunner] No trace metadata found`);
    }
    console.warn(
      `[EvalRunner] Trace input keys:`,
      trace.input ? Object.keys(trace.input) : 'no input',
    );
    return null;
  } catch (error) {
    console.error(`[EvalRunner] Error fetching trace ${traceId}:`, error);
    return null;
  }
}

/**
 * Run a single evaluation against one trace
 */
async function runSingleEval(
  draftPrompt: string,
  traceId: string,
  model: string,
  provider: ModelProvider,
  evalTraceId: string,
  options?: {
    reasoningEffort?: 'minimal' | 'low' | 'medium' | 'high';
    verbosity?: 'low' | 'medium' | 'high';
    tools?: OpenAITool[];
    originalPromptName?: string;
    // Direct observation data - avoids re-fetching if provided
    observationId?: string;
    originalOutput?: any;
  },
): Promise<TraceEvalResult> {
  const startTime = Date.now();

  try {
    // Fetch trace and extract variables from metadata.promptVariables
    // Pass originalPromptName to find the correct observation for original output
    const traceData = await fetchTraceWithVariables(
      traceId,
      options?.originalPromptName,
    );

    if (!traceData) {
      return {
        traceId,
        success: false,
        error:
          'No promptVariables found in trace metadata. Ensure generations log promptVariables in their metadata.',
      };
    }

    const { variables, originalOutput, source, schema: metadataSchema, tools: metadataTools } = traceData;

    // Use schema and tools from observation metadata if available,
    // otherwise fall back to explicitly provided tools (no schema fallback since schemas come from metadata)
    const schema = metadataSchema;
    const tools = metadataTools || options?.tools;
    
    if (metadataSchema) {
      console.log(`[EvalRunner] Using schema from observation metadata for trace ${traceId}`);
    }
    if (metadataTools) {
      console.log(`[EvalRunner] Using ${metadataTools.length} tools from observation metadata for trace ${traceId}`);
    }

    // Use directly provided original output if available (from traceObservationPairs),
    // otherwise use the one fetched from the observation
    const originalProductionOutput = options?.originalOutput
      ? typeof options.originalOutput === 'string'
        ? options.originalOutput
        : JSON.stringify(options.originalOutput)
      : originalOutput;

    if (options?.originalOutput) {
      console.log(
        `[EvalRunner] Using directly provided original output for trace ${traceId}`,
      );
    }

    // Log variables found for debugging
    console.log(
      `[EvalRunner] Found ${Object.keys(variables).length} variables for trace ${traceId}:`,
      Object.keys(variables),
    );

    // Check for common variable patterns in the template
    const templateVariableMatches = draftPrompt.match(/\{\{(\w+)\}\}/g) || [];
    const templateVariables = templateVariableMatches.map((m) =>
      m.replace(/[{}]/g, ''),
    );
    const missingVariables = templateVariables.filter(
      (v) =>
        !(v in variables) &&
        !(v.replace(/-/g, '_') in variables) &&
        !(v.replace(/_/g, '-') in variables),
    );

    if (missingVariables.length > 0) {
      console.warn(
        `[EvalRunner] Template references variables not found in trace ${traceId}:`,
        missingVariables,
      );
      console.warn(`[EvalRunner] Available variables:`, Object.keys(variables));
    }

    // Parse and fill the prompt template with variables
    const filledPrompt = parseAndFillPrompt(draftPrompt, variables);
    const promptFormat = filledPrompt.format;

    // Log a sample of the filled prompt to verify variables were replaced
    if ('messages' in filledPrompt && filledPrompt.messages.length > 0) {
      const sampleMessage = filledPrompt.messages[0];
      const hasUnfilledVariables =
        sampleMessage.content.includes('{{') ||
        sampleMessage.content.includes('{');
      if (hasUnfilledVariables) {
        console.warn(
          `[EvalRunner] Warning: Filled prompt still contains variable placeholders for trace ${traceId}`,
        );
        console.warn(
          `[EvalRunner] Sample content:`,
          sampleMessage.content.substring(0, 200),
        );
      }
    } else if ('text' in filledPrompt) {
      const hasUnfilledVariables =
        filledPrompt.text.includes('{{') || filledPrompt.text.includes('{');
      if (hasUnfilledVariables) {
        console.warn(
          `[EvalRunner] Warning: Filled prompt still contains variable placeholders for trace ${traceId}`,
        );
        console.warn(
          `[EvalRunner] Sample content:`,
          filledPrompt.text.substring(0, 200),
        );
      }
    }

    let output: string | undefined;
    let parsed: Record<string, any> | null = null;
    let tokenUsage:
      | { input?: number; output?: number; total?: number }
      | undefined;
    let toolCalls:
      | Array<{ id: string; name: string; arguments: Record<string, any> }>
      | undefined;

    if (provider === 'openai') {
      // Determine messages: use chat format if available, otherwise single user message
      const messages =
        'messages' in filledPrompt
          ? filledPrompt.messages
          : [{ role: 'user' as const, content: filledPrompt.text }];

      const response = await langfuseWrappedOpenAI({
        model,
        messages,
        // Pass schema for structured output if provided
        schema,
        // Pass tools for tool-calling if provided
        tools,
        langfuseParams: {
          traceId: evalTraceId,
          generationName: `eval-${traceId}`,
          sessionId: traceId,
          // Pass promptVariables in metadata for reference, but the filled messages are logged as input
          promptVariables: variables,
        },
        customProperties: {
          promptFormat,
          messageCount: messages.length,
          hasSchema: !!schema,
          schemaSource: metadataSchema ? 'observation-metadata' : 'none',
          hasTools: !!tools,
          toolsSource: metadataTools ? 'observation-metadata' : (options?.tools ? 'provided' : 'none'),
          toolCount: tools?.length,
        },
        // OpenAI reasoning parameters (for GPT-5)
        ...(options?.reasoningEffort
          ? { reasoning_effort: options.reasoningEffort }
          : {}),
        ...(options?.verbosity ? { verbosity: options.verbosity } : {}),
      });

      output = response?.text;
      parsed = response?.parsed ?? null;

      // Extract tool calls from response if model chose to call tools
      if (response?.toolCalls && response.toolCalls.length > 0) {
        toolCalls = response.toolCalls.map((tc: any) => ({
          id: tc.id || uuidv4(),
          name: tc.function?.name || tc.name || 'unknown',
          arguments:
            typeof tc.function?.arguments === 'string'
              ? JSON.parse(tc.function.arguments)
              : tc.function?.arguments || tc.arguments || {},
        }));
      }
    } else if (provider === 'gemini') {
      // For Gemini, convert messages to a single prompt string if chat format
      const promptText =
        'messages' in filledPrompt
          ? filledPrompt.messages
              .map((m) => `[${m.role.toUpperCase()}]\n${m.content}`)
              .join('\n\n')
          : filledPrompt.text;

      // Convert Zod schema to Gemini's native Schema format for enforcement
      const geminiSchema = schema ? zodToGeminiSchema(schema) : undefined;

      // Convert OpenAI tools to Gemini format if tools are provided
      const geminiTools = tools
        ? openaiToolsToGeminiTools(tools)
        : undefined;

      const response = await langfuseWrappedGemini({
        model,
        prompt: promptText,
        // Pass converted schema for Gemini to enforce at generation time
        schema: geminiSchema,
        // Pass converted tools for Gemini tool-calling
        tools: geminiTools,
        langfuseParams: {
          traceId: evalTraceId,
          generationName: `eval-${traceId}`,
          sessionId: traceId,
          metadata: {
            hasSchema: !!geminiSchema,
            schemaSource: metadataSchema ? 'observation-metadata' : 'none',
            schemaEnforced: !!geminiSchema,
            hasTools: !!geminiTools,
            toolsSource: metadataTools ? 'observation-metadata' : (options?.tools ? 'provided' : 'none'),
            toolCount: tools?.length,
            // Include promptVariables in metadata for Gemini (since it doesn't support promptVariables param directly)
            promptVariables: variables,
          },
        },
      });

      output = response?.text;
      if (response?.usageMetadata) {
        tokenUsage = {
          input: response.usageMetadata.promptTokenCount,
          output: response.usageMetadata.candidatesTokenCount,
          total: response.usageMetadata.totalTokenCount,
        };
      }

      // Extract tool calls from Gemini response if model chose to call tools
      if (response?.functionCalls && response.functionCalls.length > 0) {
        toolCalls = response.functionCalls.map((fc: FunctionCall) => ({
          id: fc.id || uuidv4(),
          name: fc.name || 'unknown',
          arguments: (fc.args as Record<string, any>) || {},
        }));
      }

      // Also validate with Zod for consistency (Gemini enforces, Zod validates types)
      if (schema && output) {
        try {
          const rawParsed = JSON.parse(output);
          const result = schema.safeParse(rawParsed);
          parsed = result.success ? result.data : null;
        } catch {
          // Not valid JSON, leave parsed as null
        }
      }
    }

    const latencyMs = Date.now() - startTime;

    return {
      traceId,
      success: true,
      input: variables,
      output,
      originalProductionOutput: originalProductionOutput || undefined,
      parsed,
      latencyMs,
      variableSource: source,
      tokenUsage,
      promptFormat,
      schemaKey: metadataSchema ? 'from-observation-metadata' : undefined,
      toolGroupKey: metadataTools ? 'from-observation-metadata' : undefined,
      toolCalls,
    };
  } catch (error) {
    const latencyMs = Date.now() - startTime;
    return {
      traceId,
      success: false,
      error: error instanceof Error ? error.message : String(error),
      latencyMs,
    };
  }
}

/**
 * Inngest function to run evaluations against historical traces.
 *
 * Takes a draft prompt template and a list of trace IDs, loads the traces,
 * extracts input variables from their metadata, and runs the draft prompt
 * against each trace in parallel.
 */
export const runTracesEvalFunction = inngest.createFunction(
  {
    id: 'run-traces-eval',
    concurrency: {
      limit: 5, // Max concurrent eval runs
    },
  },
  { event: 'eval/run-traces.requested' },
  async ({ event, step, logger }) => {
    const {
      draftPrompt,
      traceIds,
      traceObservationPairs,
      model,
      provider,
      evalName,
      originalPromptName,
      originalPromptVersion,
      concurrency = 10,
      langfuseTraceId,
      reasoningEffort,
      verbosity,
      schemaKey: providedSchemaKey,
      toolGroupKey: providedToolGroupKey,
    } = event.data as EvalRunInput;

    // Build a lookup map from traceId to observation data for direct access
    const observationLookup = new Map<
      string,
      { observationId: string; output?: any }
    >();
    if (traceObservationPairs) {
      for (const pair of traceObservationPairs) {
        observationLookup.set(pair.traceId, {
          observationId: pair.observationId,
          output: pair.output,
        });
      }
      logger.info(
        `[EvalRunner] Using ${traceObservationPairs.length} trace-observation pairs for direct lookup`,
      );
    }

    // NOTE: Schema and tools are now extracted from observation metadata automatically
    // The providedSchemaKey and providedToolGroupKey are kept for backward compatibility
    // but are no longer used to look up schemas/tools from a registry
    
    const evalId = uuidv4();
    const startedAt = new Date().toISOString();

    logger.info(`[EvalRunner] Starting eval run`, {
      evalId,
      evalName,
      traceCount: traceIds.length,
      model,
      provider,
      originalPromptName,
      originalPromptVersion,
      langfuseTraceId,
      reasoningEffort,
      verbosity,
      note: 'Schemas and tools will be extracted from observation metadata',
    });

    // Create a parent trace for this eval run (use pre-generated ID if provided)
    const evalTraceId = await step.run('create-eval-trace', async () => {
      const langfuse = getLangfuse();
      const traceId = langfuseTraceId || uuidv4();

      langfuse?.trace({
        id: traceId,
        name: `eval-run-${evalName}`,
        metadata: {
          evalId,
          evalName,
          originalPromptName,
          originalPromptVersion,
          traceCount: traceIds.length,
          model,
          provider,
          reasoningEffort,
          verbosity,
          note: 'Schemas and tools extracted from observation metadata',
        },
        input: {
          draftPrompt: draftPrompt.slice(0, 1000), // Truncate for logging
          traceIds,
        },
      });

      return traceId;
    });

    // Run evaluations in batches for concurrency control
    const results = await step.run('run-evaluations', async () => {
      const allResults: TraceEvalResult[] = [];

      // Process in batches
      for (let i = 0; i < traceIds.length; i += concurrency) {
        const batch = traceIds.slice(i, i + concurrency);

        logger.info(
          `[EvalRunner] Processing batch ${Math.floor(i / concurrency) + 1}`,
          {
            batchSize: batch.length,
            progress: `${i}/${traceIds.length}`,
          },
        );

        // Run batch concurrently
        const batchResults = await Promise.all(
          batch.map((traceId) => {
            // Look up observation data if available (from traceObservationPairs)
            const obsData = observationLookup.get(traceId);
            return runSingleEval(
              draftPrompt,
              traceId,
              model,
              provider,
              evalTraceId,
              {
                reasoningEffort,
                verbosity,
                originalPromptName,
                // Pass observation data directly if available
                observationId: obsData?.observationId,
                originalOutput: obsData?.output,
              },
            );
          }),
        );

        allResults.push(...batchResults);
      }

      return allResults;
    });

    // Calculate summary
    const completedAt = new Date().toISOString();
    const successCount = results.filter((r) => r.success).length;
    const failureCount = results.filter((r) => !r.success).length;

    const evalResult: EvalRunResult = {
      evalId,
      evalName,
      totalTraces: traceIds.length,
      successCount,
      failureCount,
      results,
      startedAt,
      completedAt,
      durationMs:
        new Date(completedAt).getTime() - new Date(startedAt).getTime(),
    };

    // Store detailed results in blob storage
    const resultsUrl = await step.run('store-results-blob', async () => {
      const blobPayload = {
        metadata: {
          evalId,
          evalName,
          originalPromptName,
          originalPromptVersion,
          model,
          provider,
          reasoningEffort,
          verbosity,
          note: 'Schemas and tools extracted from observation metadata',
          totalTraces: traceIds.length,
          successCount,
          failureCount,
          startedAt,
          completedAt,
          durationMs: evalResult.durationMs,
          langfuseTraceId: evalTraceId,
          // Note: results array contains originalProductionOutput field with raw model responses
        },
        // Full results with input/output/parsed/toolCalls for each trace
        results: results.map((r) => ({
          traceId: r.traceId,
          success: r.success,
          input: r.input,
          // New eval output - raw text response from the model with the draft prompt
          output: r.output,
          // Original production output - fetched from the trace being evaluated
          originalProductionOutput: r.originalProductionOutput,
          parsed: r.parsed, // Structured output if schema was used
          toolCalls: r.toolCalls, // Tool calls if tools were available
          error: r.error,
          latencyMs: r.latencyMs,
          variableSource: r.variableSource,
          tokenUsage: r.tokenUsage,
          promptFormat: r.promptFormat,
          schemaKey: r.schemaKey,
          toolGroupKey: r.toolGroupKey,
        })),
      };

      const blobKey = `eval-results-${evalName}-${evalId}`;
      const url = await storeDataInBlob(blobKey, blobPayload);

      logger.info(`[EvalRunner] Results stored in blob`, {
        evalId,
        resultsUrl: url,
      });

      return url;
    });

    // Add results URL to the eval result
    evalResult.resultsUrl = resultsUrl;

    // Update the Langfuse trace with results
    await step.run('update-eval-trace', async () => {
      const langfuse = getLangfuse();
      langfuse?.trace({
        id: evalTraceId,
        output: {
          successCount,
          failureCount,
          durationMs: evalResult.durationMs,
          resultsUrl,
        },
        metadata: {
          completed: true,
          note: 'Schemas and tools extracted from observation metadata',
          resultsUrl,
        },
      });
      await langfuse?.flushAsync();
    });

    logger.info(`[EvalRunner] Eval run completed`, {
      evalId,
      evalName,
      successCount,
      failureCount,
      durationMs: evalResult.durationMs,
      resultsUrl,
    });

    return evalResult;
  },
);
