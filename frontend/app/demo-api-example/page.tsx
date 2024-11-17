'use client'; // Add this line at the top of your file

import React, { useState, useEffect } from 'react';

const ChatBot = () => {
    const [message, setMessage] = useState('');
    const [reply, setReply] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [convoId, setConvoId] = useState<string | null>(null);
    const [messages, setMessages] = useState([]); // Store previous messages

    const handleSendMessage = async () => {
        if (!message) return;

        const requestBody = { message, convoId: convoId || undefined };

        try {
            const response = await fetch('https://hackutd-backend.vercel.app/postMessage', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            const data = await response.json();
            setReply(data.reply);
            setSuggestions(data.suggestions[0] || []); // Assuming suggestions is an array of arrays

            if (data.convoId && !convoId) {
                setConvoId(data.convoId);
            }

            // Add the new message and reply to the message history
            setMessages((prevMessages) => [
                ...prevMessages,
                { text: message, type: 'user' },
                { text: data.reply, type: 'bot' },
            ]);

            setMessage(''); // Clear the input field
        } catch (error) {
            console.error('Error sending message:', error);
        }
    };

    const handleSuggestionSelect = (event) => {
        const selectedSuggestion = event.target.value;
        setMessage(selectedSuggestion); // Set the selected suggestion as the message
    };

    useEffect(() => {
        const interval = setInterval(async () => {
            if (convoId) {
                try {
                    const response = await fetch('https://hackutd-backend.vercel.app/postMessage', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: '', convoId }),
                    });

                    const data = await response.json();
                    setReply(data.reply);
                    setSuggestions(data.suggestions[0] || []); // Assuming suggestions is an array of arrays
                } catch (error) {
                    console.error('Error polling updates:', error);
                }
            }
        }, 5000);

        return () => clearInterval(interval);
    }, [convoId]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px' }}>
            <h3>ChatBot</h3>

            {/* Display the chat history */}
            <div style={{ marginBottom: '20px', textAlign: 'left', width: '100%', padding: '0 20px' }}>
                <div>
                    {messages.map((message, index) => (
                        <div key={index} style={{ marginBottom: '10px' }}>
                            <strong>{message.type === 'user' ? 'You: ' : 'Bot: '}</strong>
                            <span>{message.text}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Suggestions table */}
            {suggestions.length > 0 && (
                <div style={{ marginTop: '20px', width: '100%' }}>
                    <strong>Suggestions:</strong>
                    <table style={{ width: '100%', border: '1px solid black', marginTop: '10px' }}>
                        <thead>
                        <tr>
                            <th></th>
                        </tr>
                        </thead>
                        <tbody>
                        {suggestions.map((suggestion, index) => (
                            <tr key={index}>
                                <td>{suggestion.name || suggestion}</td> {/* Display the suggestion text */}
                            </tr>
                        ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Input area for messages */}
            <div style={{ textAlign: 'center' }}>
                <textarea
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Type your message..."
                    style={{
                        width: '300px',
                        height: '100px',
                        textAlign: 'center',
                        color: 'black',
                        padding: '10px',
                        marginBottom: '10px',
                    }}
                />
                <br />
                <button onClick={handleSendMessage} style={{ padding: '10px 20px', cursor: 'pointer' }}>
                    Send
                </button>
            </div>
        </div>
    );
};

export default ChatBot;
