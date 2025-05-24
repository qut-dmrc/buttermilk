import { error } from '@sveltejs/kit';
import { readFile } from 'fs/promises';
import { join } from 'path';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ params }) => {
  const docFile = params.docFile;
  
  // Only allow markdown files with specific names for security
  if (!['app-integration.md', 'static-deployment.md'].includes(docFile)) {
    throw error(404, 'Documentation file not found');
  }
  
  try {
    // Read the markdown file from lib directory
    const filePath = join(process.cwd(), 'src', 'lib', docFile);
    const content = await readFile(filePath, 'utf-8');
    
    return {
      content,
      title: docFile.replace('.md', '').split('-').map((word: string) => 
        word.charAt(0).toUpperCase() + word.slice(1)
      ).join(' ')
    };
  } catch (err) {
    console.error('Error loading documentation:', err);
    throw error(500, 'Could not load documentation');
  }
};
