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
    <div className="flex flex-col h-screen bg-background">

      {/* Chat Header */}
      <div className="bg-secondary border-b border-border p-4 shadow-sm">
        <CardTitle className="text-lg font-semibold text-secondary-foreground">First-Steps AI Therapy</CardTitle>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="p-6 flex flex-col space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={cn(
                  "flex w-full max-w-2xl rounded-lg",
                  message.sender === 'user' ? 'justify-start' : 'justify-end'
                )}
              >
                {message.sender === 'bot' ? (
                  <>
                    <Avatar className="mr-3 h-10 w-10">
                      <AvatarImage src={botAvatarUrl} alt="AI Avatar"/>
                      <AvatarFallback>AI</AvatarFallback>
                    </Avatar>
                    <Card className="w-fit bg-accent text-left shadow-md">
                      <CardContent className="p-4">{message.text}</CardContent>
                    </Card>
                  </>
                ) : (
                  <>
                    <Card className="w-fit bg-primary text-right text-primary-foreground shadow-md">
                      <CardContent className="p-4">{message.text}</CardContent>
                    </Card>
                    <Avatar className="ml-3 h-10 w-10">
                      <AvatarImage src={userAvatarUrl} alt="User Avatar"/>
                      <AvatarFallback>US</AvatarFallback>
                    </Avatar>
                  </>
                )}
              </div>
            ))}
            <div ref={scrollRef} />
          </div>
        </ScrollArea>
      </div>

      {/* Chat Input */}
      <div className="bg-secondary border-t border-border p-4">
        <div className="flex items-center space-x-3">
          <Textarea
            ref={inputRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleInputKeyDown}
            placeholder="Type your message..."
            className="flex-1 resize-none shadow-sm rounded-md"
            rows={1}
          />
          <Button onClick={handleSend} className="shadow-sm rounded-md">Send</Button>
        </div>
      </div>
    </div>
  );
}
