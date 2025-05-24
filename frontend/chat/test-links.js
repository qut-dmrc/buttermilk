/**
 * Test script to verify URL and DOI link detection in terminal messages
 * Run this with: node test-links.js
 */

// Function to simulate sending a message with URLs and DOIs to the terminal
function testMessageWithLinks() {
  console.log("Test 1: Basic URL formats:");
  console.log("Check out this website: https://example.com");
  console.log("This URL starts with www: www.example.org/path/to/resource");
  console.log("URL with query parameters: https://api.example.com/data?id=123&type=json");
  
  console.log("\nTest 2: DOI formats:");
  console.log("Here's a DOI: 10.1234/abcd.5678");
  console.log("DOI with doi: prefix: doi:10.1038/nature09410");
  console.log("DOI with https://doi.org/ prefix: https://doi.org/10.1093/nar/gkaa942");
  
  console.log("\nTest 3: Mixed formats in a paragraph:");
  console.log("In this paper (doi:10.1038/nature09410), the authors demonstrate a novel approach. You can also check their website at https://research-lab.example.edu for more information. Another related study is available at 10.1093/nar/gkaa942.");
  
  console.log("\nTest 4: URLs and DOIs in different contexts:");
  console.log("- Command output with URL: Result saved to www.example.com/results/data.json");
  console.log("- Error message referencing DOI: Could not locate resource 10.1234/abcd.5678");
  console.log("- Log entry with both: [INFO] User accessed https://api.example.org and requested document doi:10.1038/nature09410");
}

// Call the test function
testMessageWithLinks();
