import { render, screen } from '@testing-library/react';
import App from './App';

test('renders ECE Header', () => {
  render(<App />);
  const linkElement = screen.getByText(/ECE470/i);
  expect(linkElement).toBeInTheDocument();
});
