export function safeSubstring(str: string | undefined, start: number, end: number): string {
  if (str === undefined) {
    return '';
  }
  return str.substring(start, end);
}
