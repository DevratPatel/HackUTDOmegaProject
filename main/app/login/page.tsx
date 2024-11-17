'use client';

import React, { useState } from 'react';
import { signInWithGoogle, signInWithEmail } from '../../firebase';

export default function LoginPage() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleEmailLogin = (e: React.FormEvent) => {
        e.preventDefault();
        signInWithEmail(email, password); // Call the Firebase email/password login function
    };

    return (
        <div className="flex h-screen">
            {/* Left Side */}
            <div className="w-1/2 bg-gray-100 flex items-center justify-center">
                <div className="text-center">
                    <div className="rounded-full border-4 border-gray-300 w-40 h-40 mx-auto"></div>
                    <p className="mt-4 text-gray-600 text-lg">Logo</p>
                </div>
            </div>

            {/* Right Side */}
            <div className="w-1/2 flex flex-col justify-center items-center bg-white">
                <form onSubmit={handleEmailLogin} className="w-2/3 max-w-md">
                    <div className="mb-4">
                        <label htmlFor="username" className="block text-gray-700 mb-2">
                            Username
                        </label>
                        <input
                            id="username"
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="Enter your username"
                            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
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
                            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
                        />
                    </div>
                    <button
                        type="submit"
                        className="w-full bg-blue-500 text-white py-2 rounded-md shadow-md hover:bg-blue-600"
                    >
                        Login
                    </button>
                </form>
                <button
                    onClick={signInWithGoogle}
                    className="mt-4 bg-red-500 text-white px-6 py-2 rounded-md shadow-md hover:bg-red-600"
                >
                    Login with Google
                </button>
                <div className="mt-4 text-center">
                    <p className="text-gray-600 text-sm">
                        Don’t have an account?{' '}
                        <a href="/signup" className="text-blue-500 hover:underline">
                            Sign up
                        </a>
                    </p>
                    <p className="text-gray-600 text-sm mt-2">
                        Forgot your password?{' '}
                        <a href="/reset-password" className="text-blue-500 hover:underline">
                            Reset here
                        </a>
                    </p>
                </div>
            </div>
        </div>
    );
}
