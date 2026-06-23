// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore"; 
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyABozdo9_PCfnGhe6Z9w-khHTjh1GAahRQ",
  authDomain: "agrosaarthi-9191b.firebaseapp.com",
  projectId: "agrosaarthi-9191b",
  storageBucket: "agrosaarthi-9191b.firebasestorage.app",
  messagingSenderId: "926102784768",
  appId: "1:926102784768:web:ce19984c0efdaf9e8e81e5"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);