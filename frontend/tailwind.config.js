/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#f3f1eb",
        ink: "#1b1e1b",
        pine: "#1f4a38",
        moss: "#e7f0ea",
        line: "#e2dfd6",
        muted: "#5f675f",
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        sans: ["Outfit", "Segoe UI", "sans-serif"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(27, 30, 27, 0.04), 0 12px 32px rgba(27, 30, 27, 0.04)",
      },
    },
  },
  plugins: [],
};
