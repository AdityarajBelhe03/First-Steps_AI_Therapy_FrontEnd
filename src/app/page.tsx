"use client";

import React, {useState, useRef, useEffect} from 'react';
import {Textarea} from "@/components/ui/textarea";
import {Button} from "@/components/ui/button";
import {Card, CardContent, CardHeader, CardTitle} from "@/components/ui/card";
import {Avatar, AvatarImage, AvatarFallback} from "@/components/ui/avatar";
import {ScrollArea} from "@/components/ui/scroll-area";
import {cn} from "@/lib/utils";
import {generateEmpatheticResponse} from '@/ai/flows/generate-empathetic-response';

const userAvatarUrl = `https://picsum.photos/id/237/36/36`;
const botAvatarUrl = `https://picsum.photos/id/888/36/36`;

interface Message {
  sender: 'user' | 'bot';
  text: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    scrollToBottom();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    scrollRef.current?.scrollIntoView({behavior: "smooth", block: "end", inline: "nearest"});
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {sender: 'user', text: input};
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput('');

    // Call the AI flow
    try {
      const aiResponse = await generateEmpatheticResponse({
        message: input,
        chatHistory: messages.map(m => `${m.sender}: ${m.text}`).join('\n'),
      });

      const botResponse: Message = {
        sender: 'bot',
        text: aiResponse.response,
      };
      setMessages(prevMessages => [...prevMessages, botResponse]);
    } catch (error) {
      console.error("Failed to generate AI response:", error);
      const botResponse: Message = {
        sender: 'bot',
        text: "Sorry, I'm having trouble generating a response right now. Please try again later.",
      };
      setMessages(prevMessages => [...prevMessages, botResponse]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">

      {/* Chat Header */}
      <div className="bg-white border-b border-gray-200 p-4">
        <CardTitle className="text-lg font-semibold">Empath AI</CardTitle>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="p-4 flex flex-col space-y-2">
            {messages.map((message, index) => (
              <div
                key={index}
                className={cn(
                  "flex w-full max-w-md mb-2",
                  message.sender === 'user' ? 'justify-end' : 'justify-start'
                )}
              >
                {message.sender === 'bot' && (
                  <Avatar className="mr-2">
                    <AvatarImage src={botAvatarUrl} alt="AI Avatar"/>
                    <AvatarFallback>AI</AvatarFallback>
                  </Avatar>
                )}
                <Card className={cn(
                  "w-fit rounded-lg",
                  message.sender === 'user' ? 'bg-blue-100 text-right' : 'bg-green-100 text-left'
                )}>
                  <CardContent className="p-3">{message.text}</CardContent>
                </Card>
                {message.sender === 'user' && (
                  <Avatar className="ml-2">
                    <AvatarImage src={userAvatarUrl} alt="User Avatar"/>
                    <AvatarFallback>US</AvatarFallback>
                  </Avatar>
                )}
              </div>
            ))}
            <div ref={scrollRef} />
          </div>
        </ScrollArea>
      </div>

      {/* Chat Input */}
      <div className="bg-white border-t border-gray-200 p-4">
        <div className="flex items-center space-x-2">
          <Textarea
            ref={inputRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleInputKeyDown}
            placeholder="Type your message..."
            className="flex-1 resize-none shadow-sm"
            rows={1}
          />
          <Button onClick={handleSend} className="shadow-sm">Send</Button>
        </div>
      </div>
    </div>
  );
}
