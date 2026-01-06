import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { trainingFile, model, suffix, hyperparameters } = body;

    if (!trainingFile || typeof trainingFile !== 'string') {
      return NextResponse.json(
        { error: 'Training file URL is required' },
        { status: 400 },
      );
    }

    if (!model || typeof model !== 'string') {
      return NextResponse.json(
        { error: 'Base model is required' },
        { status: 400 },
      );
    }

    // Download the training file from blob storage
    let trainingFileBuffer: Buffer;
    try {
      const response = await fetch(trainingFile);
      if (!response.ok) {
        throw new Error(
          `Failed to fetch training file: ${response.statusText}`,
        );
      }
      const arrayBuffer = await response.arrayBuffer();
      trainingFileBuffer = Buffer.from(arrayBuffer);
    } catch (error) {
      return NextResponse.json(
        {
          error: `Failed to fetch training file: ${error instanceof Error ? error.message : 'Unknown error'}`,
        },
        { status: 400 },
      );
    }

    // Upload the training file to OpenAI
    // Create a File-like object from the buffer for Node.js environment
    // Convert Buffer to Uint8Array to satisfy TypeScript BlobPart type
    const uint8Array = new Uint8Array(trainingFileBuffer);
    const blob = new Blob([uint8Array], {
      type: 'application/x-ndjson',
    });
    const file = new File([blob], 'training.jsonl', {
      type: 'application/x-ndjson',
    });

    // Upload file to OpenAI
    const uploadedFile = await openai.files.create({
      file: file,
      purpose: 'fine-tune',
    });

    // Wait for file to be processed
    let fileStatus = uploadedFile.status;
    while (fileStatus !== 'processed' && fileStatus !== 'error') {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const fileInfo = await openai.files.retrieve(uploadedFile.id);
      fileStatus = fileInfo.status;
    }

    if (fileStatus !== 'processed') {
      return NextResponse.json(
        {
          error: `File processing failed with status: ${fileStatus}`,
        },
        { status: 400 },
      );
    }

    // Start the fine-tuning job
    const fineTuneParams: any = {
      training_file: uploadedFile.id,
      model: model,
    };

    if (suffix) {
      fineTuneParams.suffix = suffix;
    }

    if (hyperparameters) {
      const hyperparams: any = {};
      if (hyperparameters.batch_size !== undefined) {
        hyperparams.batch_size = hyperparameters.batch_size;
      }
      if (hyperparameters.learning_rate_multiplier !== undefined) {
        hyperparams.learning_rate_multiplier =
          hyperparameters.learning_rate_multiplier;
      }
      if (hyperparameters.n_epochs !== undefined) {
        hyperparams.n_epochs = hyperparameters.n_epochs;
      }
      if (Object.keys(hyperparams).length > 0) {
        fineTuneParams.hyperparameters = hyperparams;
      }
    }

    const fineTuneJob = await openai.fineTuning.jobs.create(fineTuneParams);

    return NextResponse.json({
      success: true,
      id: fineTuneJob.id,
      status: fineTuneJob.status,
      model: fineTuneJob.model,
      createdAt: fineTuneJob.created_at,
      trainingFile: uploadedFile.id,
    });
  } catch (error) {
    console.error('Error starting fine-tuning job:', error);
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : 'Failed to start fine-tuning job',
      },
      { status: 500 },
    );
  }
}
