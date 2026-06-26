export function ema(values: number[], length: number): number[] {
  if (values.length === 0) return [];
  const multiplier = 2 / (length + 1);
  const output: number[] = [values[0]];

  for (let i = 1; i < values.length; i += 1) {
    output.push((values[i] - output[i - 1]) * multiplier + output[i - 1]);
  }

  return output;
}
