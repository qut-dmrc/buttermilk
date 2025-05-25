#!/bin/bash
# Script to clean and rebuild dependencies

echo "Cleaning node_modules..."
rm -rf node_modules

echo "Cleaning .svelte-kit cache..."
rm -rf .svelte-kit

echo "Cleaning Vite cache..."
rm -rf node_modules/.vite

echo "Reinstalling dependencies..."
npm install

echo "All done! Try running the development server with 'npm run dev'"
