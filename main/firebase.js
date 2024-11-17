// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth, signInWithEmailAndPassword, GoogleAuthProvider, signInWithPopup, sendPasswordResetEmail } from 'firebase/auth';
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyC_W_k5PbX0XW18C70ZOfo0UjKRLVt0170",
  authDomain: "hackutdomegaproject-5725f.firebaseapp.com",
  projectId: "hackutdomegaproject-5725f",
  storageBucket: "hackutdomegaproject-5725f.firebasestorage.app",
  messagingSenderId: "651141126049",
  appId: "1:651141126049:web:313d51d0b08bceffdedde0"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// Google Sign-In
const googleProvider = new GoogleAuthProvider();
const signInWithGoogle = async () => {
    try {
        const result = await signInWithPopup(auth, googleProvider);
        console.log("User logged in:", result.user);
        alert(`Welcome, ${result.user.displayName}`);
    } catch (error) {
        console.error("Error logging in with Google:", error.message);
        alert("Login failed. Please try again.");
    }
};

// Email and Password Sign-In
const signInWithEmail = async (email, password) => {
    try {
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        console.log("User logged in:", userCredential.user);
        alert("Login successful!");
    } catch (error) {
        console.error("Error logging in with email/password:", error.message);
        alert("Invalid email or password.");
    }
};

const resetPassword = async (email) => {
    try {
        await sendPasswordResetEmail(auth, email);
        console.log('Password reset email sent successfully');
      } 
    catch (error) {
        console.error('Error sending password reset email:', error.message);
        throw error; 
    }
}

export { auth, signInWithGoogle, signInWithEmail, resetPassword };