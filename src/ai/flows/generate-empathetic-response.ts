// src/ai/flows/generate-empathetic-response.ts
'use server';

/**
 * @fileOverview Generates empathetic and therapeutic responses to user messages.
 *
 * - generateEmpatheticResponse - A function that generates an empathetic response.
 * - GenerateEmpatheticResponseInput - The input type for the generateEmpatheticResponse function.
 * - GenerateEmpatheticResponseOutput - The return type for the generateEmpatheticResponse function.
 */

import {ai} from '@/ai/ai-instance';
import {z} from 'genkit';
import { classifyEmotion, EmotionResult } from '@/services/emotion-classifier';

const GenerateEmpatheticResponseInputSchema = z.object({
  message: z.string().describe('The user message to respond to.'),
  chatHistory: z.string().describe('The previous chat history.'),
});
export type GenerateEmpatheticResponseInput = z.infer<typeof GenerateEmpatheticResponseInputSchema>;

const GenerateEmpatheticResponseOutputSchema = z.object({
  response: z.string().describe('The empathetic and therapeutic response.'),
});
export type GenerateEmpatheticResponseOutput = z.infer<typeof GenerateEmpatheticResponseOutputSchema>;

export async function generateEmpatheticResponse(input: GenerateEmpatheticResponseInput): Promise<GenerateEmpatheticResponseOutput> {
  return generateEmpatheticResponseFlow(input);
}

const emotionClassificationTool = ai.defineTool({
  name: 'classifyEmotion',
  description: 'Classifies the emotion present in a given text.',
  inputSchema: z.object({
    text: z.string().describe('The input text to analyze for emotion.'),
  }),
  outputSchema: z.object({
    label: z.string().describe('The label of the detected emotion (e.g., \'joy\', \'sadness\', \'anger\').'),
    score: z.number().describe('The confidence score associated with the detected emotion, ranging from 0 to 1.'),
  }),
  async resolve(input) {
    const emotionResult = await classifyEmotion(input.text);
    return {
      label: emotionResult.label,
      score: emotionResult.score,
    };
  },
});

const prompt = ai.definePrompt({
  name: 'generateEmpatheticResponsePrompt',
  tools: [emotionClassificationTool],
  input: {
    schema: z.object({
      message: z.string().describe('The user message to respond to.'),
      chatHistory: z.string().describe('The previous chat history.'),
    }),
  },
  output: {
    schema: z.object({
      response: z.string().describe('The empathetic and therapeutic response.'),
    }),
  },
  prompt: `You are a kind and compassionate AI therapy assistant. A user has sent you the following message: {{{message}}}.  Here is the chat history: {{{chatHistory}}}.

Classify the emotion of the user's message using the classifyEmotion tool.  Based on the emotion expressed in the user's message, generate an empathetic and therapeutic response.

Respond in a way that demonstrates you understand their feelings and are there to support them.
`,
});

const generateEmpatheticResponseFlow = ai.defineFlow<
  typeof GenerateEmpatheticResponseInputSchema,
  typeof GenerateEmpatheticResponseOutputSchema
>({
  name: 'generateEmpatheticResponseFlow',
  inputSchema: GenerateEmpatheticResponseInputSchema,
  outputSchema: GenerateEmpatheticResponseOutputSchema,
},
async input => {
  const {output} = await prompt(input);
  return output!;
});
