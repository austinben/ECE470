import './App.css';

function App() {
  function startModel() {
    console.log("clicked");
  }
  return (
    <div className="App">
      <header className="App-header">
        <h3>
          ECE 470 - Artificial Intelligence - Brain Tumour Detection
        </h3>
      </header>
      <div className="App-Body">
        <p>
          Click to train model! 
        </p>
        <button className="App-Button" onClick={startModel}>
          Cool AI Button
        </button>
      </div>
    </div>
  );
}

export default App;
