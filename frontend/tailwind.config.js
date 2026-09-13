/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
        "./public/**/*.html",
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                mono: ['JetBrains Mono', 'monospace'],
                'display-hero': ['Plus Jakarta Sans', 'sans-serif'],
                'headline-md': ['Plus Jakarta Sans', 'sans-serif'],
                'headline-lg': ['Plus Jakarta Sans', 'sans-serif'],
                'headline-xl': ['Plus Jakarta Sans', 'sans-serif'],
                'label-code': ['JetBrains Mono', 'monospace'],
                'label-mono-xs': ['JetBrains Mono', 'monospace'],
                'body-lg': ['Inter', 'sans-serif'],
                'body-md': ['Inter', 'sans-serif'],
                'body-sm': ['Inter', 'sans-serif'],
            },
            colors: {
                brand: {
                    50: '#ecfdf5',
                    100: '#d1fae5',
                    200: '#a7f3d0',
                    300: '#6ee7b7',
                    400: '#34d399',
                    500: '#10b981',
                    600: '#059669',
                    700: '#047857',
                    800: '#065f46',
                    900: '#064e3b',
                    950: '#022c22',
                },
            },
            animation: {
                'fade-in': 'fadeIn 0.3s ease-out',
                'slide-up': 'slideUp 0.3s ease-out',
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            },
            keyframes: {
                fadeIn: { from: { opacity: 0 }, to: { opacity: 1 } },
                slideUp: { from: { opacity: 0, transform: 'translateY(10px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
            },
            backdropBlur: { xs: '2px' },
        },
    },
    plugins: [
        require('@tailwindcss/forms'),
    ],
}