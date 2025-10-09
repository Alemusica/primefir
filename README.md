# primefir~ Max External

`primefir~` è un filtro FIR stereo basato su sinc finestrata progettato per l'ambiente Max/MSP. L'implementazione combina sequenze aperiodiche (come numeri primi o rapporti irrazionali) con una raccolta di finestre radiali e differenti strategie di interpolazione frazionaria per ottenere risposte in frequenza ricche e controllabili.

## Caratteristiche principali
- **Sequenze di offset**: lineare, numeri primi (incluso indice prime⊗φ e scala φ), progressioni irrazionali (φ, √2, costante plastica, π, e). La tabella dei primi viene costruita con un crivello ottimizzato e un limite superiore basato su Dusart per robustezza.
- **Finestre radiali**: Hann, Hamming, Blackman, Blackman-Harris a 4 termini, Nuttall e Kaiser, tutte mappate sulla distanza normalizzata così che il picco sia centrato nel kernel.
- **Interpolazioni configurabili**: nessuna, lineare, Lagrange4 causale, Catmull-Rom (Keys), Farrow cubico/quintico. Le rispettive latenze e margini sono calcolati automaticamente per mantenere la linearità di fase quando richiesto.
- **Normalizzazione e gain compensation**: il tap centrale più i tap simmetrici vengono sommati per calcolare il guadagno DC, evitando instabilità. È disponibile un'opzione di compensazione percepita via √freq.
- **Supporto DSP solido**: gestione del ring buffer stereo 2×, interpolazione frazionaria precomputata, latenza coerente fra modalità lineari e causali, cancellazione e reset dello stato DSP.

## Costruzione
Questo repository usa CMake e gli script del Max SDK per generare il bundle `primefir~.mxo`. Impostare la variabile `MAX_SDK_PATH` verso l'SDK installato e, opzionalmente, `MAX_PACKAGE_DIR` per la destinazione dell'installazione. Esempio di build su macOS:

```bash
cmake -S . -B build -DMAX_SDK_PATH="$HOME/max-sdk"
cmake --build build
cmake --install build
```

Il target predefinito è compilato in C++17 e include la configurazione richiesta dal Max SDK tramite gli script `max-pretarget` e `max-posttarget`.
