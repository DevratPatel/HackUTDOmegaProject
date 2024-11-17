"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { signInWithGoogle, signInWithEmail } from "../firebase";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();

  const handleEmailLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try{
      const user = await signInWithEmail(email, password); // Call the Firebase email/password login function
      console.log("Logged in user:", user);
      router.push("/dashboard");
    }
    catch(error) {
    console.error("Login failed:", error);
    alert("Invalid email or password");
    }
  };

  const handleGoogleLogin = async () => {
    try {
      const user = await signInWithGoogle();
      console.log("Logged in user:", user);
      router.push("/dashboard");
      } 
      catch (error) {
      console.error("Google login failed:", error);
      alert("Google login failed. Please try again.");
    }
  }

  return (
    <div className="flex h-screen bg-gray-900">
      {/* Left Side */}
      <div className="w-1/2 bg-gray-800 flex items-center justify-center">
        <div className="text-center">
          <div className="rounded-full border-4 border-gray-600 w-40 h-40 mx-auto"></div>
          <p className="mt-4 text-gray-300 text-lg">Logo</p>
        </div>
      </div>

      {/* Right Side */}
      <div className="w-1/2 flex flex-col justify-center items-center bg-gray-900">
        <form onSubmit={handleEmailLogin} className="w-2/3 max-w-md">
          <div className="mb-4">
            <label htmlFor="username" className="block text-gray-300 mb-2">
              Username
            </label>
            <input
              id="username"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter your username"
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-100 placeholder-gray-500"
            />
          </div>
          <div className="mb-6">
            <label htmlFor="password" className="block text-gray-300 mb-2">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-100 placeholder-gray-500"
            />
          </div>
          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-2 rounded-md shadow-md hover:bg-blue-700"
          >
            Login
          </button>
        </form>
        <button
          onClick={handleGoogleLogin}
          className="mt-4 bg-red-600 text-white px-6 py-2 rounded-md shadow-md hover:bg-red-700"
        >
          Login with Google
        </button>
        <div className="mt-4 text-center">
          <p className="text-gray-400 text-sm">
            Don't have an account?{" "}
            <a
              href="/signup"
              className="text-blue-400 hover:text-blue-300 hover:underline"
            >
              Sign up
            </a>
          </p>
          <p className="text-gray-400 text-sm mt-2">
            Forgot your password?{" "}
            <a
              href="/reset-password"
              className="text-blue-400 hover:text-blue-300 hover:underline"
            >
              Reset here
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}

