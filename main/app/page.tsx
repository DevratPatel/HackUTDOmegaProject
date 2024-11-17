"use client";

import React, { useState } from "react";
import { signInWithGoogle, signInWithEmail } from "../firebase";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();

  const handleEmailLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await signInWithEmail(email, password);
      router.push("/dashboard");
    } catch (error) {
      console.error("Login failed:", error);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      await signInWithGoogle();
      router.push("/dashboard");
    } catch (error) {
      console.error("Google login failed:", error);
    }
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-4">
      <Card className="w-full max-w-[900px] h-[500px] bg-[#0A0A0A] border-zinc-800 flex rounded-xl overflow-hidden">
        {/* Left side with logo */}
        <div className="w-1/2 flex items-center justify-center border-r border-zinc-800">
          <div className="w-32 h-32 bg-white rounded-full" />
        </div>

        {/* Right side with form */}
        <div className="w-1/2 flex flex-col justify-center px-12">
          <form onSubmit={handleEmailLogin} className="space-y-6">
            <div className="space-y-4">
              <Input
                type="email"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="bg-transparent border-zinc-800 h-12"
              />
              <Input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="bg-transparent border-zinc-800 h-12"
              />
            </div>

            <Button
              type="submit"
              className="w-full bg-white text-black hover:bg-zinc-200 h-12"
            >
              Login
            </Button>
          </form>

          <div className="mt-6 space-y-4">
            <Button
              onClick={handleGoogleLogin}
              variant="outline"
              className="w-full border-zinc-800 h-12 space-x-2"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path
                  fill="currentColor"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="currentColor"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="currentColor"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                />
                <path
                  fill="currentColor"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                />
              </svg>
              <span>Continue with Google</span>
            </Button>

            <div className="text-sm text-zinc-400 space-y-2">
              <p>
                Don't have an account?{" "}
                <a href="/signup" className="text-white hover:underline">
                  Sign Up
                </a>
              </p>
              <p>
                Forgot your password?{" "}
                <a
                  href="/reset-password"
                  className="text-white hover:underline"
                >
                  Reset Here
                </a>
              </p>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
