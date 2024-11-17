// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import {getAuth, signInWithEmailAndPassword, GoogleAuthProvider, signInWithPopup } from 'firebase/auth';
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDgvyuU8_NtuwlJxjbB4XiJkXOiu7uKNn8",
  authDomain: "hackutdomegaproject.firebaseapp.com",
  projectId: "hackutdomegaproject",
  storageBucket: "hackutdomegaproject.firebasestorage.app",
  messagingSenderId: "489374733859",
  appId: "1:489374733859:web:f2567979117acd6b19c396",
  measurementId: "G-9MGFDZBP64"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const auth = getAuth(app);

// Google Sign-In
const googleProvider = new GoogleAuthProvider();
const signInWithGoogle = async () => {
    try {
        const result = await signInWithPopup(auth, googleProvider);
        console.log("User logged in:", result.user);
        alert(`Welcome, ${result.user.displayName}`);
    } catch (error) {
        console.error("Error logging in:", error.message);
        alert("Login failed. Please try again.");
    }
};

// Email and Password Login
const signInWithEmail = async (email, password) => {
    try {
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        console.log("User logged in:", userCredential.user);
        alert("Login successful!");
    } catch (error) {
        console.error("Error logging in:", error.message);
        alert("Invalid email or password.");
    }
};

export { auth, signInWithGoogle, signInWithEmail };