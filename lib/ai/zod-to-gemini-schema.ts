/**
 * Converts Zod schemas to Gemini's native Schema format.
 * This allows us to use the same Zod schemas for both OpenAI and Gemini evals.
 */

import { Schema, Type } from '@google/genai';
import { z } from 'zod';

/**
 * Convert a Zod schema to Gemini's Schema format
 */
export function zodToGeminiSchema(zodSchema: z.ZodSchema<any>): Schema {
  return convertZodType(zodSchema);
}

function convertZodType(schema: z.ZodTypeAny): Schema {
  const typeName = (schema._def as any).typeName;

  switch (typeName) {
    case 'ZodString':
      return handleZodString(schema as z.ZodString);

    case 'ZodNumber':
      return { type: Type.NUMBER };

    case 'ZodBoolean':
      return { type: Type.BOOLEAN };

    case 'ZodArray':
      return {
        type: Type.ARRAY,
        items: convertZodType((schema as any)._def.type),
      };

    case 'ZodObject':
      return handleZodObject(schema as z.ZodObject<any>);

    case 'ZodEnum':
      const enumValues = (schema as any)._def.values;
      return {
        type: Type.STRING,
        enum: enumValues,
      };

    case 'ZodOptional':
      return convertZodType((schema as any)._def.innerType);

    case 'ZodNullable':
      return {
        ...convertZodType((schema as any)._def.innerType),
        nullable: true,
      };

    case 'ZodDefault':
      return convertZodType((schema as any)._def.innerType);

    case 'ZodEffects':
      // Handle .describe() and other effects
      return convertZodType((schema as any)._def.schema);

    case 'ZodLiteral':
      const literalValue = (schema as any)._def.value;
      if (typeof literalValue === 'string') {
        return { type: Type.STRING, enum: [literalValue] };
      } else if (typeof literalValue === 'number') {
        return { type: Type.NUMBER };
      } else if (typeof literalValue === 'boolean') {
        return { type: Type.BOOLEAN };
      }
      return { type: Type.STRING };

    case 'ZodUnion':
      // For unions, try to find common type or use STRING
      const unionTypes = (schema as any)._def.options;
      // Check if all are strings (common for enum-like unions)
      const allStrings = unionTypes.every(
        (t: z.ZodTypeAny) =>
          (t._def as any).typeName === 'ZodLiteral' &&
          typeof (t as any)._def.value === 'string',
      );
      if (allStrings) {
        const values = unionTypes.map(
          (t: z.ZodLiteral<any>) => (t._def as any).value as string,
        );
        return { type: Type.STRING, enum: values };
      }
      // Fallback: convert first option
      return convertZodType(unionTypes[0]);

    default:
      console.warn(
        `[zodToGeminiSchema] Unknown Zod type: ${typeName}, defaulting to STRING`,
      );
      return { type: Type.STRING };
  }
}

function handleZodString(schema: z.ZodString): Schema {
  const checks = (schema._def as any).checks || [];

  // Check for enum-like constraints
  for (const check of checks) {
    if (check.kind === 'regex') {
      // Can't directly translate regex to Gemini
      return { type: Type.STRING };
    }
  }

  return { type: Type.STRING };
}

function handleZodObject(schema: z.ZodObject<any>): Schema {
  const shape = (schema._def as any).shape();
  const properties: Record<string, Schema> = {};
  const required: string[] = [];

  for (const [key, value] of Object.entries(shape)) {
    const zodValue = value as z.ZodTypeAny;
    properties[key] = convertZodType(zodValue);

    // Check if field is required (not optional)
    if (!zodValue.isOptional()) {
      required.push(key);
    }

    // Add description if available
    if ((zodValue._def as any).description) {
      properties[key].description = (zodValue._def as any).description;
    }
  }

  return {
    type: Type.OBJECT,
    properties,
    required: required.length > 0 ? required : undefined,
  };
}
