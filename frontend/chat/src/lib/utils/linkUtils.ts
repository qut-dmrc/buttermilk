/**
 * Utility functions for processing text with URLs and DOIs into clickable links
 */

/**
 * Process text content to convert URLs and DOIs to clickable links
 * @param text The text content to process
 * @returns Text with URLs and DOIs converted to HTML links
 */
export function processLinksInText(text: string): string {
  if (!text) return '';
  
  // URL regex pattern - matches most common URL formats
  // Handles http://, https://, www. and even bare domains with paths
  const urlRegex = /(https?:\/\/(?:www\.)?|www\.)[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+(?:\/[^\s]*)?/g;
  
  // DOI regex pattern - matches DOI identifiers in various formats
  // Handles formats like: 10.1234/abc123, doi:10.1234/abc123, https://doi.org/10.1234/abc123
  const doiRegex = /(?:doi:|https?:\/\/doi\.org\/)?10\.\d{4,}\/[a-zA-Z0-9.\/-]+/g;
  
  // Replace URLs with clickable links first
  let processedText = text.replace(urlRegex, (url) => {
    // Ensure URL has proper protocol prefix
    const href = url.startsWith('www.') ? 'https://' + url : url.startsWith('http') ? url : 'https://' + url;
    return `<a href="${href}" target="_blank" rel="noopener noreferrer">${url}</a>`;
  });
  
  // Then replace DOIs with clickable links
  processedText = processedText.replace(doiRegex, (doi) => {
    // Extract the DOI portion (10.xxxx/xxxx) if it includes a prefix
    const doiPart = doi.includes('10.') ? doi.substring(doi.indexOf('10.')) : doi;
    
    // Create the proper DOI URL
    const href = `https://doi.org/${doiPart}`;
    return `<a href="${href}" target="_blank" rel="noopener noreferrer">${doi}</a>`;
  });
  
  return processedText;
}
