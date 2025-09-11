import React from 'react'
import { Link } from 'react-router-dom'

export default function FAQ() {
  return (
    <div>
      <h3 style={{ marginTop: 24 }}>is this...</h3>
      <ul>
        <li>
          an official Hack Club You Ship, We Ship program? <strong>yes</strong>
        </li>
      </ul>

      <h3 style={{ marginTop: 24 }}>what is...</h3>
      <ul>
        <li>
          a neural network?
          A layered function approximator composed of simple units (“neurons”) with
          learnable weights. By stacking layers and nonlinearities, it can model complex input→output relationships.
          {' '}See
          {' '}<a href="https://en.wikipedia.org/wiki/Artificial_neural_network" target="_blank" rel="noreferrer noopener">Artificial neural network</a>
          {' '}for an overview and the
          {' '}<a href="https://en.wikipedia.org/wiki/Multilayer_perceptron" target="_blank" rel="noreferrer noopener">multi‑layer perceptron (MLP)</a>
          {' '}we use here to map observations into actions.
        </li>
        <li>
          a genetic algorithm?
          An evolutionary optimization method inspired by natural selection: keep the best performers,
          recombine them (crossover), and randomly mutate parameters over generations to discover
          higher‑fitness solutions. See
          {' '}<a href="https://en.wikipedia.org/wiki/Genetic_algorithm" target="_blank" rel="noreferrer noopener">Genetic algorithm</a>
          {' '}on Wikipedia.
        </li>
      </ul>

      <h3 style={{ marginTop: 24 }}>you must...</h3>
      <ul>
        <li>be in good standing at Hack Club and be willing to submit to identity verification for prizes</li>
        <li>open-source your project if you want to be eligible for the prize</li>
      </ul>

      <h3 style={{ marginTop: 24 }}>you may...</h3>
      <ul>
        <li>use the example project as a starting point for your submission, but it must be <em>heavily modified</em> adding impressive features.  We can hash that out in advance, before you start, upon reviewing & approving your proposal video</li>
        <li>use AI to assist the development of your project (I did!)</li>
        <li>use any supervised or unsupervised ML technique to train your neural network</li>
        <li>
          leverage existing general-use ML libraries
        </li>
      </ul>

      <h3 style={{ marginTop: 24 }}>you could...</h3>
      <ul>
        <li>build a game using <a href="https://www.pygame.org/docs/">pygame</a> that self-learns a neural network using a genetic algorithm (<Link to="/example">my example</Link>)</li>
        <li>
          create a new <a href="https://developer.gimp.org/" target="_blank" rel="noreferrer noopener">GIMP</a> filter that manipulates image content, for instance a style-aware smart brush
        </li>
        <li>
          write an <a href="https://www.audacityteam.org/" target="_blank" rel="noreferrer noopener">Audacity</a> plugin that auto-separates vocals/drums/bass
        </li>
      </ul>

    </div>
  )
}
