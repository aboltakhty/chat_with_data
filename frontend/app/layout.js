import './globals.css';

export const metadata = {
  title: 'PDF Chat',
  description: 'Chat with your PDFs using OpenAI, LangChain, and FAISS'
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="app-shell">
          {children}
        </div>
      </body>
    </html>
  );
}
