'use client';

import React, { useState } from 'react';
import { createUserWithEmailAndPassword } from 'firebase/auth';
import { auth } from '../../firebase';

export default function SignUpPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      console.log('User created:', userCredential.user);
      alert('Sign-up successful! You can now log in.');
    } catch (error) {
        if (error instanceof Error) {
            // TypeScript now knows error is an instance of Error
            console.error('Error during sign-up:', error.message);
            alert(`Sign-up failed: ${error.message}`);
        } else {
            console.error('Unknown error during sign-up:', error);
            alert('Sign-up failed due to an unknown error. Please try again.');
        }
    }
  };

  return (
    <div className="flex h-screen items-center justify-center">
      <form onSubmit={handleSignUp} className="w-2/3 max-w-md">
        <div className="mb-4">
          <label htmlFor="email" className="block text-gray-700 mb-2">
            Email
          </label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400 text-black"
          />
        </div>
        <div className="mb-6">
          <label htmlFor="password" className="block text-gray-700 mb-2">
            Password
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter your password"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400 text-black"
          />
        </div>
        <button
          type="submit"
          className="w-full bg-green-500 text-white py-2 rounded-md shadow-md hover:bg-green-600"
        >
          Sign Up
        </button>
      </form>
    </div>
  );
}
