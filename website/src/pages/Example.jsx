import React from 'react'

export default function Example() {
  return (
    <div>
      <center><h1 style={{ marginTop: 24 }}>Collect Coins For Heidi!</h1></center>
      <p>
        This simple platformer prototype showcases a compact <a href="https://en.wikipedia.org/wiki/Artificial_neural_network" target="_blank" rel="noreferrer noopener">neural network</a> policy optimized with a
        <a href="https://en.wikipedia.org/wiki/Genetic_algorithm" target="_blank" rel="noreferrer noopener"> genetic algorithm (GA)</a>.
        Agents start with random weights and evolve over generations as the GA selects the best performers,
        recombines them, and mutates parameters. The policy learns to navigate platforms, collect coins and powerups, and
        reach the goal using simple scalar feedback (fitness) from each episode.
      </p>
      <ul>
        <li>Neural network controller (MLP) deciding actions from game observations</li>
        <li>Genetic algorithm with selection, crossover, and mutation</li>
        <li>Fitness signals: coins, finish bonus, (optionally) progress, and small jump costs</li>
        <li>Deterministic replays and checkpoints for inspecting generations</li>
      </ul>
      <p> Here is an example of gameplay:</p>
      <video
        controls
        playsInline
        muted
        loop
        autoPlay
        poster="/img/gameplay.png"
        style={{ width: '50%', display: 'block', margin: '0 auto 16px', borderRadius: 12 }}
      >
        <source src="https://hc-cdn.hel1.your-objectstorage.com/s/v3/241065c6422a7c3b168d069a47040fc93dbf697e_demo.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>
      
      <h3 style={{ marginTop: 24 }}>what feedback did the model get?</h3>
      <ul>
        <li><strong>Forward progress</strong>: reward for moving toward the goal</li>
        <li><strong>Coin collection</strong>: reward when picking up coins</li>
        <li><strong>Finish bonus</strong>: reward for reaching the raccoon; extra if faster</li>
        <li><strong>Jump cost</strong>: small penalty each time a jump is initiated</li>
        <li><strong>Failure penalty</strong>: large penalty if time runs out, you fall, or exit bounds</li>
        <li><strong>Aggregation</strong>: fitness is total episode reward (averaged across eval runs)</li>
      </ul>
      <h3 style={{ marginTop: 24 }}>maps and generalization</h3>
      <img src="/img/map.png" alt="Map editor preview" style={{ width: '50%', borderRadius: 12, display: 'block', margin: '8px auto 12px' }} />
      <p>
        The level is driven by an easy-to-edit ASCII map. Adding or tweaking platforms, coins, and entities is fast,
        which makes iterating on environments simple. One obvious next step would be to add <em>multi-map training</em> — i.e. training across
        a diverse map-set so the policy learns to generalize and can handle new layouts it hasn't seen before.
      </p>

      <h3 style={{ marginTop: 24 }}>training the model</h3>
      <img src="/img/train.png" alt="Training logs preview" style={{ width: '50%', borderRadius: 12, display: 'block', margin: '8px auto 12px' }} />
      <p>
        During training, thousands of competing games are simulated in parallel using mutated variants of the policy.
        Progress is tracked by watching for new best-performers (see <em>max</em> in the logs) and the average population
        performance (see <em>mean</em>). The mutation rate factor (<em>sigma</em>) is annealed over time: it starts higher to
        encourage broad exploration and narrows later to refine and converge to more stable optima.
      </p>

      <h3 style={{ marginTop: 24 }}>contrail snapshots</h3>
      <div className="contrail-gallery">
        <div className="contrail-item" style={{ textAlign: 'center' }}>
          <img src="/img/first_gen.png" alt="First generation contrail" style={{ borderRadius: 12 }} />
          <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>First gen: no learning yet; follows the goal-only hint.</div>
        </div>
        <div className="contrail-item" style={{ textAlign: 'center' }}>
          <img src="/img/stuck.png" alt="Stuck contrail" style={{ borderRadius: 12 }} />
          <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>Stuck: learned to grab boots and found the coin chamber; explores hotspot.</div>
        </div>
        <div className="contrail-item" style={{ textAlign: 'center' }}>
          <img src="/img/success.png" alt="Successful contrail" style={{ borderRadius: 12 }} />
          <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>Success: collects many coins and delivers them to Heidi at the goal.</div>
        </div>
      </div>

      <h3 style={{ marginTop: 24 }}>neural network architecture</h3>
      <ul>
        <li><strong>Inputs</strong>: flattened observation vector from the environment.</li>
        <li><strong>Hidden layers</strong>: two fully connected layers of 32 units each (configurable).</li>
        <li><strong>Activation</strong>: tanh between hidden layers for smooth, bounded nonlinearity.</li>
        <li><strong>Bias</strong>: handled by appending a constant 1 to each layer’s input before the weight multiply.</li>
        <li><strong>Outputs</strong>: linear logits over discrete actions; the chosen action is the argmax.</li>
        <li><strong>Parameters</strong>: all weights (incl. biases) packed into one flat vector for efficient GA crossover/mutation.</li>
      </ul>

      <h3 style={{ marginTop: 24 }}>libraries used</h3>
      <ul>
        <li><strong>pygame</strong>: realtime rendering, input handling, assets, and windowing.</li>
        <li><strong>NumPy</strong>: neural-network math (vectorized MLP forward pass), parameter storage/mutation.</li>
        <li><strong>watchdog</strong> (optional): file watching for live checkpoint playback with <code>--watch</code>.</li>
      </ul>

      <h3 style={{ marginTop: 24 }}>raycasting</h3>
      <div className="contrail-gallery">
        <div className="contrail-item" style={{ textAlign: 'center' }}>
          <img src="/img/raycast1.png" alt="Raycast step 1" />
          <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>A coin is detected top-right.  Simple radial raycasting is performed every few frames.</div>
        </div>
        <div className="contrail-item" style={{ textAlign: 'center' }}>
          <img src="/img/raycast2.png" alt="Raycast step 2" />
          <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>A coin is detected in front and behind.  On frames where raycasts aren’t sampled, prior results are reused along with a separate recency score.</div>
        </div>
        <div className="contrail-item" style={{ textAlign: 'center' }}>
          <img src="/img/raycast3.png" alt="Raycast step 3" />
          <div style={{ fontSize: 12, color: '#555', marginTop: 6 }}>A coin is detected down and to the left.  Awareness of immediate surroundings helps the model reach higher fitness in fewer generations and improves generality.</div>
        </div>
      </div>
    </div>
  )
}
