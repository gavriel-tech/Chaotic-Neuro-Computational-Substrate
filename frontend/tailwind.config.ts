import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#00ff99",
        accent: "#00cc77",
        highlight: "#ccffee",
        inactive: "#1a1a1a",
        bg: {
          primary: "#000000",
          secondary: "#0a0a12",
        },
      },
      boxShadow: {
        "glow-sm": "0 0 8px rgba(0, 255, 153, 0.4)",
        "glow-md": "0 0 12px rgba(0, 255, 153, 0.5)",
        "glow-lg": "0 0 20px rgba(0, 255, 153, 0.6)",
        "glow-xl": "0 0 10px rgba(0, 204, 119, 0.5), inset 0 0 10px rgba(0, 255, 153, 0.1)",
      },
      animation: {
        glitch: "glitch 3s infinite",
        "pulse-glow": "pulse-glow 2s infinite",
        scanline: "scanline 8s linear infinite",
      },
      keyframes: {
        glitch: {
          "0%, 90%, 100%": { transform: "translate(0)" },
          "92%": { transform: "translate(-2px, 2px)" },
          "94%": { transform: "translate(2px, -2px)" },
          "96%": { transform: "translate(-2px, -2px)" },
          "98%": { transform: "translate(2px, 2px)" },
        },
        "pulse-glow": {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.8", transform: "scale(0.95)" },
        },
        scanline: {
          "0%": { transform: "translateY(0)" },
          "100%": { transform: "translateY(4px)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
