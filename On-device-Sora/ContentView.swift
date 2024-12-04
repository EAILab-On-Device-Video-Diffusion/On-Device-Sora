import SwiftUI

struct ContentView: View {
    @State var isGenerating: Bool = false
    
    @State var prompt: String = "a beautiful waterfall"
    @State private var seed = "42"
    @State private var aestheticScore = "6.5"
    @State private var step = 30
    @State private var mergeStep = 15
    @State private var numLpltarget = 15

    @State private var isLPL = false
    @State private var isTDTM = false
    @State private var isCI = false
    @State private var isDL = false
  
    @StateObject private var tensor2vidConverter = Tensor2Vid()
  
    var body: some View {
        List {
          VStack(alignment: .leading) {
            Text("Prompt:")
            TextField("Enter prompt,but default exists", text: $prompt).padding(4).background(Color(uiColor: .secondarySystemBackground))
          }
          HStack {
            VStack(alignment: .leading) {
                        Text("LPL").font(.system(size: 22, weight: .semibold)).lineLimit(2)
                        HStack {
                            if self.isLPL {
                                Text("On")
                            } else {
                                Text("Off")
                            }
                            Spacer()
                            Toggle("", isOn: $isLPL)
                        }
                    }
                    .frame(width: 100)
                    .padding()
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(self.isLPL ? Color.green: Color.gray, lineWidth: 2)
                    )
            VStack(alignment: .leading) {
                        Text("TDTM").font(.system(size: 22, weight: .semibold)).lineLimit(2)
                        HStack {
                          if self.isTDTM {
                                Text("On")
                            } else {
                                Text("Off")
                            }
                            Spacer()
                          Toggle("", isOn: $isTDTM)
                        }
                    }
                    .frame(width: 100)
                    .padding()
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                          .stroke(self.isTDTM ? Color.green: Color.gray, lineWidth: 2)
                    )
          }
          HStack{
            VStack(alignment: .leading) {
                        Text("CI").font(.system(size: 22, weight: .semibold)).lineLimit(2)
                        HStack {
                          if self.isCI {
                                Text("On")
                            } else {
                                Text("Off")
                            }
                            Spacer()
                            Toggle("", isOn: $isCI)
                        }
                    }
                    .frame(width: 100)
                    .padding()
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(self.isCI ? Color.green: Color.gray, lineWidth: 2)
                    )
            VStack(alignment: .leading) {
                        Text("DL").font(.system(size: 22, weight: .semibold)).lineLimit(2)
                        HStack {
                            if self.isDL {
                                Text("On")
                            } else {
                                Text("Off")
                            }
                            Spacer()
                            Toggle("", isOn: $isDL)
                        }
                    }
                    .frame(width: 100)
                    .padding()
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(self.isDL ? Color.green: Color.gray, lineWidth: 2)
                    )
          }
          VStack(alignment: .leading) {
            Text("Seed:")
            TextField("42", text: $seed).keyboardType(.decimalPad).padding(4).background(Color(uiColor: .secondarySystemBackground))
          }
          
          VStack(alignment: .leading) {
            Text("Aesthetic score:")
            TextField("6.5", text: $aestheticScore).keyboardType(.decimalPad).padding(4).background(Color(uiColor: .secondarySystemBackground))
          }
          
          Stepper(
            value: $step,
            in: 0...50,
            step: 1
          ) {
            Text("Iteration steps: \(step)")
          }
          Stepper(
            value: $mergeStep,
            in: 0...step,
            step: 1
          ) {
            Text("Merge steps: \(mergeStep)")
          }
          
          Stepper(
            value: $numLpltarget,
            in: 0...50,
            step: 1
          ) {
            Text("LPL target steps: \(numLpltarget)")
          }
          
          if isGenerating {
            if let videoURL = tensor2vidConverter.videoURL {
                VideoPlayerView(url: videoURL)
            } else {
              ProgressView()
            }
          }
          
          Button(action: generate) {
            Text("Start Video Generate").font(.title)
          }.buttonStyle(.borderedProminent)
        }
        .padding()
    }
  
  func generate() {
      do {
        isGenerating = true
        let soraPipeline = try SoraPipeline(resourcesAt: Bundle.main.bundleURL, videoConverter: tensor2vidConverter)
        print("Start Video Generate")
        let aesprompt = prompt.appending(" aesthetic score: \(aestheticScore).")
        let logdir = FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask)[0]
        soraPipeline.sample(prompt: aesprompt, logdir: logdir, seed: Int(seed) ?? 42, step: step, mergeStep: mergeStep, numLpltarget: numLpltarget, isLPL: isLPL, isTDTM: isTDTM, isCI: isCI, isDL: isDL)
      } catch {
          print("Error: Can't initiallize SoraPipeline")
      }
    }
}
