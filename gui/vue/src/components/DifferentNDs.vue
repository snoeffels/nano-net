<template>
  <v-container>
    <v-row>
      <v-col cols="12">
        <v-stepper
            v-model="e6"
            vertical
        >

          <!-- STEP 1 -->
          <v-stepper-step
              :complete="e6 > 1"
              step="1"
          >
            Select inputs
            <small>Select the path to the folders with your input images. Every folder can only contain images of one condition.</small>
          </v-stepper-step>
          <v-stepper-content step="1">
            <p>All whitespace or non alphanumerical characters will be removed from labels. Please specify at least two
              inputs with unique names to continue.</p>

            <v-row>
              <v-col>
                <fieldset
                    style="border: 1px rgba(113,113,113,0.4) solid; border-radius: 5px; padding-left: 10px; padding-right: 20px">

                  <legend style="margin-left: 24px; padding: 0.2em 0.8em; color: rgba(0,0,0,0.6);font-size: 12px">Input
                    1
                  </legend>
                  <div style="display:flex">
                    <v-text-field
                        v-model="conditionName"
                        @change="handleNameChange"
                        label="Name"
                        style="padding-left: 33px; padding-top: 20px; width: 100%"
                    />

                    <input
                       v-model="color"
                       type="color"
                       style="width: 33px;margin-top: 34px;padding-left: 10px"
                    />
                  </div>

                  <v-file-input
                      @click.prevent="selectInputFolder"
                      @click:clear="fileInputValue = null;"
                      label="Input"
                      :value="fileInputValue"
                      style="padding-top: 0;margin-top: -3px"
                  />

                </fieldset>
              </v-col>
              <v-col>
                <fieldset
                    style="border: 1px rgba(113,113,113,0.4) solid; border-radius: 5px; padding-left: 10px; padding-right: 20px">

                  <legend style="margin-left: 24px; padding: 0.2em 0.8em; color: rgba(0,0,0,0.6);font-size: 12px">Input
                    2
                  </legend>

                  <div style="display:flex">
                    <v-text-field
                        v-model="condition2Name"
                        @change="handleNameChange2"
                        label="Name"
                        style="padding-left: 33px; padding-top: 20px; width: 100%"
                    />

<!--                    <input-->
<!--                        v-model="color2"-->
<!--                        type="color"-->
<!--                        style="width: 33px;margin-top: 34px;padding-left: 10px"-->
<!--                    />-->
                  </div>


                  <v-file-input
                      @click.prevent="selectInputFolder2"
                      @click:clear="fileInput2Value = null;"
                      label="Input"
                      :value="fileInput2Value"
                      style="padding-top: 0;margin-top: -3px"
                  />
                </fieldset>

              </v-col>
            </v-row>
            <v-row class="pb-5">
              <v-col>
                <fieldset
                    style="border: 1px rgba(113,113,113,0.4) solid; border-radius: 5px; padding-left: 10px; padding-right: 20px">

                  <legend style="margin-left: 24px; padding: 0.2em 0.8em; color: rgba(0,0,0,0.6);font-size: 12px">Input
                    3
                  </legend>
                  <div style="display:flex">
                    <v-text-field
                        v-model="condition3Name"
                        @change="handleNameChange3"
                        label="Name"
                        style="padding-left: 33px; padding-top: 20px; width: 100%"
                    />

<!--                    <input-->
<!--                        v-model="color3"-->
<!--                        type="color"-->
<!--                        style="width: 33px;margin-top: 34px;padding-left: 10px"-->
<!--                    />-->
                  </div>

                  <v-file-input
                      @click.prevent="selectInputFolder3"
                      @click:clear="fileInput3Value = null;"
                      label="Input"
                      :value="fileInput3Value"
                      style="padding-top: 0;margin-top: -3px"
                  />
                </fieldset>

              </v-col>
              <v-col>
                <fieldset
                    style="border: 1px rgba(113,113,113,0.4) solid; border-radius: 5px; padding-left: 10px; padding-right: 20px">

                  <legend style="margin-left: 24px; padding: 0.2em 0.8em; color: rgba(0,0,0,0.6);font-size: 12px">Input
                    4
                  </legend>
                  <div style="display:flex">
                    <v-text-field
                        v-model="condition4Name"
                        @change="handleNameChange4"
                        label="Name"
                        style="padding-left: 33px; padding-top: 20px; width: 100%"
                    />

<!--                    <input-->
<!--                        v-model="color4"-->
<!--                        type="color"-->
<!--                        style="width: 33px;margin-top: 34px;padding-left: 10px"-->
<!--                    />-->
                  </div>

                  <v-file-input
                      @click.prevent="selectInputFolder4"
                      @click:clear="fileInput4Value = null;"
                      label="Input"
                      :value="fileInput4Value"
                      style="padding-top: 0;margin-top: -3px"
                  />
                </fieldset>


              </v-col>
            </v-row>


            <v-btn
                small
                color="primary"
                @click="handleNext"
                :disabled="[fileInputValue, fileInput2Value, fileInput3Value, fileInput4Value].filter(e => e !== null).length < 2"
            >
              Next
            </v-btn>
            <v-btn text small @click="$emit('back')" class="ml-3">
              Home
            </v-btn>
          </v-stepper-content>

          <!-- STEP 2 -->
          <v-stepper-step
              :complete="e6 > 2"
              step="2"
          >
            Order of conditions
            <small>Determines in which order the conditions will appear in the various plots at the end.</small>
          </v-stepper-step>
          <v-stepper-content step="2">
            <v-row
                align="center"
            >
              <v-col cols="12">
                <v-autocomplete
                    v-model="globals.order"
                    :items="orderItems"
                    dense
                    chips
                    small-chips
                    multiple
                    style="width: 280px"
                />
              </v-col>
            </v-row>


            <v-btn
                small
                color="primary"
                @click="handleNext2"
                :disabled="globals.order.length !== orderItems.length"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 1" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 3 -->
          <v-stepper-step
              :complete="e6 > 3"
              step="3"
          >
            Parameter settings
            <small>Set parameters for the Random Forest, t-SNE, and K-NN</small>
          </v-stepper-step>
          <v-stepper-content step="3">
            <v-row>
              <v-col cols="6">
                <h4>
                  Random Forest
                </h4>
                <v-row>
                  <v-col>
                    <v-text-field
                        v-model="globals.testSize"
                        label="Train/Test split"
                        hint="e.g. 0.2 means the RF will train on 80% of the data and test with 20%"
                        type="number"
                        step="0.01"
                    />
                    <v-text-field
                        v-model="globals.nEstimators"
                        label="N-Estimators"
                        hint="The number of trees in the forest"
                        type="number"
                        step="1"
                    />
                    <v-text-field
                        v-model="globals.minSamplesSplit"
                        label="Minimum samples split"
                        hint="The minimum number of samples required to split an internal node (default: 2)"
                        type="number"
                        step="1"
                    />
                  </v-col>
                  <v-col>
                    <v-text-field
                        v-model="globals.minSamplesLeaf"
                        label="Minimum leaf samples"
                        hint="The minimum number of samples required to be at a leaf node (default: 1)"
                        type="number"
                        step="1"
                    />
                    <v-text-field
                        v-model="globals.maxDepth"
                        label="Maximal tree depth"
                        hint="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples."
                        type="number"
                        step="1"
                    />
                  </v-col>
                </v-row>
              </v-col>
              <v-col cols="3">
                <h4>
                  t-SNE
                </h4>
                <v-text-field
                    v-model="globals.nComponents"
                    label="N-Components"
                    hint="Dimension of the embedded space (default: 2)"
                    type="number"
                    step="0.01"
                />
                <v-text-field
                    v-model="globals.perplexity"
                    label="Perplexity"
                    hint="Consider selecting a value between 5 and 50. Different values can result in significantly different results. The perplexity must be less that the number of samples."
                    type="number"
                    step="1"
                />
                <v-text-field
                    v-model="globals.nIter"
                    label="N-Iterations"
                    hint="Maximum number of iterations for the optimization. Should be at least 250"
                    type="number"
                    step="1"
                />
              </v-col>
              <v-col cols="3">
                <h4>
                  K-NN
                </h4>
                <v-text-field
                    v-model="globals.testSizeKnn"
                    label="Train/Test split"
                    hint="e.g. 0.2 means K-NN will train on 80% of the data and test with 20%"
                    type="number"
                    step="0.01"
                />
                <v-text-field
                    v-model="globals.nNeighbors"
                    label="Maximal number of neighbors"
                    type="number"
                    hint="e.g. 10 will evaluate the performance of the classifier with each k = 1, 2, 3, ... , 10"
                    step="1"
                />
              </v-col>
            </v-row>
            <v-btn
                small
                color="primary"
                @click="handleNext3"
                :disabled="false"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 2" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 4 -->
          <v-stepper-step
              :complete="e6 > 4"
              step="4"
          >
            Estimation of K-NN optimal number
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="4">
            <v-text-field
                v-model="globals.acuracyKnn"
                label="K-NN Acuracy"
                type="number"
                step="0.01"
                readonly
                style="width: 150px"
            />
            <v-text-field
                v-model="globals.optimalNumberOfNeighbors"
                label="Optimal number of neighbors"
                type="number"
                step="1"
                readonly
                style="width: 150px"
            />

            <v-btn
                small
                color="primary"
                @click="handleNext4"
                :disabled="false"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 3" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 5 -->
          <v-stepper-step
              :complete="e6 > 5"
              step="5"
          >
            Select features
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="5">
            <v-row>
              <v-col>
                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.area"
                    @change="setFeatures('area')"
                    hint="Test 1 2 3"
                    checked
                    label="ND area (microns)"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.meanArea"
                    @change="setFeatures('mean_area')"
                    label="Mean ND area per image"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.varArea"
                    @change="setFeatures('var_area')"
                    label="Variance of ND area per image"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.density"
                    @change="setFeatures('density')"
                    label="ND density"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.intensity"
                    @change="setFeatures('intensity')"
                    label="ND intensity"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.relativeIntensity"
                    @change="setFeatures('relative_intensity')"
                    label="Relative ND intensity"
                />
              </v-col>
              <v-col>
                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.meanIntensity"
                    @change="setFeatures('mean_intensity')"
                    label="Mean ND intensity per image"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.varIntensity"
                    @change="setFeatures('var_intensity')"
                    label="Variance of ND intensity per image"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.maxIntensity"
                    @change="setFeatures('max_intensity')"
                    label="ND max. intensity"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.meanEccentricity"
                    @change="setFeatures('mean_eccentricity')"
                    label="Mean ND eccentricity per image"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.equivalentDiameterArea"
                    @change="setFeatures('equivalent_diameter_area')"
                    label="ND equivalent diameter area"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.meanEquivalentDiameterArea"
                    @change="setFeatures('mean_equivalent_diameter_area')"
                    label="Mean ND equivalent diameter area per image"
                />
              </v-col>
              <v-col>
                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.perimeter"
                    @change="setFeatures('perimeter')"
                    checked
                    label="Perimeter of NDs"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.meanPerimeter"
                    @change="setFeatures('meanPerimeter')"
                    label="Mean perimeter of NDs per image"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.ndQuantity"
                    @change="setFeatures('nano_domain_quantity')"
                    label="Number of NDs per image"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.SCI"
                    @change="setFeatures('sci')"
                    label="SCI"
                />

                <v-checkbox
                    class="ml-2"
                    v-model="checkboxes.densityMicrons"
                    @change="setFeatures('density_microns')"
                    label="ND density in microns"
                />
              </v-col>
            </v-row>

            <v-btn
                small
                color="primary"
                @click="handleNext5"
                :disabled="false"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 4" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 6 -->
          <v-stepper-step
              :complete="e6 > 6"
              step="6"
          >
          Output path
            <small>Select a path to an output folder where results will be saved.</small>
          </v-stepper-step>
          <v-stepper-content step="6">
            <v-file-input
                @click.prevent="selectOutputFolder"
                @click:clear="fileOutputValue = null"
                label="Output"
                :value="fileOutputValue"
            />
            <v-btn
                small
                color="primary"
                @click="e6 = 7;run()"
                :disabled="!fileOutputValue"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 5" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 7 -->
          <v-stepper-step step="7">
            Save results
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="7">

            <v-progress-linear
                readonly
                :value="progress"
                height="15"
            >
              <strong style="color: white; font-size: 10px">{{ progressCurrentStep }} / {{ progressSteps }}</strong>
            </v-progress-linear>

            <v-tabs
                v-model="tabs"
                background-color="primary"
                dark
            >
              <v-tab>
                Boxplots
              </v-tab>

              <v-tab>
                t-SNE
              </v-tab>

              <v-tab>
                K-NN
              </v-tab>

              <v-tab>
                RF
              </v-tab>

              <v-dialog
                  v-model="showReport"
                  fullscreen
                  hide-overlay
                  transition="dialog-bottom-transition"
              >
                <template v-slot:activator="{ on }">
                  <div class="v-tab ml-auto" v-on="on">
                    Report
                  </div>
                </template>

                <v-card style="background-color: black">
                  <v-toolbar
                      dark
                      color="primary"
                  >
                    <v-btn
                        icon
                        dark
                        @click="showReport = false"
                    >
                      <v-icon>mdi-close</v-icon>
                    </v-btn>
                    <v-toolbar-title>
                      Report
                    </v-toolbar-title>
                  </v-toolbar>

                  <p style="color: white; overflow: auto" class="pa-2" v-html="report" />
                </v-card>
              </v-dialog>

            </v-tabs>

            <v-tabs-items v-model="tabs">
              <v-tab-item>
                <v-card flat>
                  <div style="display: flex;overflow-x: auto;" class="mt-5">
                    <div v-for="image in images.boxplot" :key="image.title" style="margin: 0 0 0 5px">
                      <v-dialog
                          v-model="image.show"
                          fullscreen
                          hide-overlay
                          transition="dialog-bottom-transition"
                      >
                        <template v-slot:activator="{ on }">
                          <button v-on="on">
                            <v-card style="border-radius: 3px;width: 100px">
                              <v-img :src="srcFromImage(image.src)"/>
                            </v-card>
                          </button>

                        </template>
                        <v-card style="background-color: black">
                          <v-toolbar
                              dark
                              color="primary"
                          >
                            <v-btn
                                icon
                                dark
                                @click="image.show = false"
                            >
                              <v-icon>mdi-close</v-icon>
                            </v-btn>
                            <v-toolbar-title>
                              {{ image.title }}
                            </v-toolbar-title>
                          </v-toolbar>

                          <v-img :src="srcFromImage(image.src)" style="max-height: calc(100vh - 56px);max-width: 100vw"/>
                        </v-card>
                      </v-dialog>
                    </div>
                  </div>
                </v-card>
              </v-tab-item>

              <v-tab-item>
                <v-card flat>
                  <div style="display: flex;overflow-x: auto;" class="mt-5">
                    <div v-for="image in images.tsne" :key="image.title" style="margin: 0 0 0 5px">
                      <v-dialog
                          v-model="image.show"
                          fullscreen
                          hide-overlay
                          transition="dialog-bottom-transition"
                      >
                        <template v-slot:activator="{ on }">
                          <button v-on="on">
                            <v-card style="border-radius: 3px;width: 100px">
                              <v-img :src="srcFromImage(image.src)"/>
                            </v-card>
                          </button>

                        </template>
                        <v-card style="background-color: black">
                          <v-toolbar
                              dark
                              color="primary"
                          >
                            <v-btn
                                icon
                                dark
                                @click="image.show = false"
                            >
                              <v-icon>mdi-close</v-icon>
                            </v-btn>
                            <v-toolbar-title>
                              {{ image.title }}
                            </v-toolbar-title>
                          </v-toolbar>

                          <v-img :src="srcFromImage(image.src)" style="max-height: calc(100vh - 56px);max-width: 100vw"/>
                        </v-card>
                      </v-dialog>
                    </div>
                  </div>
                </v-card>
              </v-tab-item>

              <v-tab-item>
                <v-card flat>
                  <div style="display: flex;overflow-x: auto;" class="mt-5">
                    <div v-for="image in images.knn" :key="image.title" style="margin: 0 0 0 5px">
                      <v-dialog
                          v-model="image.show"
                          fullscreen
                          hide-overlay
                          transition="dialog-bottom-transition"
                      >
                        <template v-slot:activator="{ on }">
                          <button v-on="on">
                            <v-card style="border-radius: 3px;width: 100px">
                              <v-img :src="srcFromImage(image.src)"/>
                            </v-card>
                          </button>

                        </template>
                        <v-card style="background-color: black">
                          <v-toolbar
                              dark
                              color="primary"
                          >
                            <v-btn
                                icon
                                dark
                                @click="image.show = false"
                            >
                              <v-icon>mdi-close</v-icon>
                            </v-btn>
                            <v-toolbar-title>
                              {{ image.title }}
                            </v-toolbar-title>
                          </v-toolbar>

                          <v-img :src="srcFromImage(image.src)" style="max-height: calc(100vh - 56px);max-width: 100vw"/>
                        </v-card>
                      </v-dialog>
                    </div>
                  </div>
                </v-card>
              </v-tab-item>

              <v-tab-item>
                <v-card flat>
                  <div style="display: flex;overflow-x: auto;" class="mt-5">
                    <div v-for="image in images.rf" :key="image.title" style="margin: 0 0 0 5px">
                      <v-dialog
                          v-model="image.show"
                          fullscreen
                          hide-overlay
                          transition="dialog-bottom-transition"
                      >
                        <template v-slot:activator="{ on }">
                          <button v-on="on">
                            <v-card style="border-radius: 3px;width: 100px">
                              <v-img :src="srcFromImage(image.src)"/>
                            </v-card>
                          </button>

                        </template>
                        <v-card style="background-color: black">
                          <v-toolbar
                              dark
                              color="primary"
                          >
                            <v-btn
                                icon
                                dark
                                @click="image.show = false"
                            >
                              <v-icon>mdi-close</v-icon>
                            </v-btn>
                            <v-toolbar-title>
                              {{ image.title }}
                            </v-toolbar-title>
                          </v-toolbar>

                          <v-img :src="srcFromImage(image.src)" style="max-height: calc(100vh - 56px);max-width: 100vw"/>
                        </v-card>
                      </v-dialog>
                    </div>
                  </div>
                </v-card>
              </v-tab-item>
            </v-tabs-items>

            <v-row>
              <v-col class="mt-5">
                <v-btn
                    small
                    color="primary"
                    @click="run"
                    :disabled="!done"
                >
                  Rerun
                </v-btn>
<!--                <v-btn outlined small color="error" class="ml-3" @click="cancel" :disabled="done">-->
<!--                  Cancel-->
<!--                </v-btn>-->
                <v-btn text small @click="e6 = 6" class="ml-3" :disabled="!done">
                  Back
                </v-btn>
                <v-btn text small @click="$emit('back')" class="ml-3" :disabled="!done">
                  Home
                </v-btn>
              </v-col>
            </v-row>
          </v-stepper-content>
        </v-stepper>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: 'DifferentNDs',
  data() {
    return {
      e6: 1,
      fileInputValue: null,
      path: "",
      color: '#fd8f00',
      conditionName: "Label1",
      fileInput2Value: null,
      path2: "",
      color2: '#0063b9',
      condition2Name: "Label2",
      fileInput3Value: null,
      path3: "",
      color3: '#4b9d00',
      condition3Name: "Label3",
      fileInput4Value: null,
      path4: "",
      color4: '#da00d7',
      condition4Name: "Label4",
      fileOutputValue: null,
      features: ['area', 'mean_area', 'var_area', 'density', 'intensity', 'relative_intensity', 'mean_intensity', 'var_intensity', 'max_intensity', 'mean_eccentricity', 'equivalent_diameter_area', 'mean_equivalent_diameter_area', 'perimeter', 'mean_perimeter', 'nano_domain_quantity', 'sci', 'density_microns'],
      orderItems: [],
      pathItems: [],
      images: {
        boxplot: [],
        tsne: [],
        knn: [],
        rf: [],
      },
      progress: 0,
      progressCurrentStep: 0,
      progressSteps: 0,
      tabs: null,
      done: false,
      checkboxes: {
        area: true,
        meanArea: true,
        varArea: true,
        density: true,
        intensity: true,
        relativeIntensity: true,
        meanIntensity: true,
        varIntensity: true,
        maxIntensity: true,
        meanEccentricity: true,
        equivalentDiameterArea: true,
        meanEquivalentDiameterArea: true,
        perimeter: true,
        meanPerimeter: true,
        ndQuantity: true,
        SCI: true,
        densityMicrons: true,
      },
      showReport: false,
      report: "",
      globals: {
        order: [],
        paths: [],
        testSize: 0.2,
        nEstimators: 1000,
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
        maxDepth: 110,
        testSizeKnn: 0.2,
        nNeighbors: 3,
        nComponents: 2,
        perplexity: 20,
        nIter: 500,
        acuracyKnn: 0.0,
        optimalNumberOfNeighbors: 0,
      }
    }
  },
  methods: {
    srcFromImage(image) {
      return 'data:image/png;base64,' + image
    },
    handleNext() {
      this.globals.order = [];
      this.orderItems = [];

      if (this.fileInputValue) {
        this.orderItems.push(this.conditionName)
        this.pathItems.push(this.path)
        this.globals.order.push(this.conditionName)

        if (typeof eel !== 'undefined') {
          // eslint-disable-next-line no-undef
          eel.set_name_and_color(this.conditionName, this.color);
        }
      }

      if (this.fileInput2Value) {
        this.orderItems.push(this.condition2Name)
        this.pathItems.push(this.path2)
        this.globals.order.push(this.condition2Name)

        if (typeof eel !== 'undefined') {
          // eslint-disable-next-line no-undef
          eel.set_name_and_color_2(this.condition2Name, this.color2);
        }
      }

      if (this.fileInput3Value) {
        this.orderItems.push(this.condition3Name)
        this.pathItems.push(this.path3);
        this.globals.order.push(this.condition3Name)

        if (typeof eel !== 'undefined') {
          // eslint-disable-next-line no-undef
          eel.set_name_and_color_3(this.condition3Name, this.color3);
        }
      }

      if (this.fileInput4Value) {
        this.orderItems.push(this.condition4Name)
        this.pathItems.push(this.path4)
        this.globals.order.push(this.condition4Name)

        if (typeof eel !== 'undefined') {
          // eslint-disable-next-line no-undef
          eel.set_name_and_color_4(this.condition4Name, this.color4);
        }
      }

      this.e6 = 2;
    },
    handleNext2() {
      this.globals.paths = [];

      for (var i = 0; i < this.globals.order.length; i++) {
        this.globals.paths.push(this.pathItems[this.orderItems.indexOf(this.globals.order[i])]);
      }

      if (typeof eel !== 'undefined') {
        // eslint-disable-next-line no-undef
        eel.set_conditions_and_paths(this.globals.order, this.globals.paths);

      }

      this.e6 = 3;
    },
    handleNext3() {
      if (typeof eel !== 'undefined') {
        // eslint-disable-next-line no-undef
        eel.set_parameters(this.globals);

        var that = this;
        // eslint-disable-next-line no-undef
        eel.get_accuracy()(function (values) {
          console.log("accuracy: " + values['accuracy']);
          console.log("neighbors: " + values['neighbors']);
          that.globals.acuracyKnn = values['accuracy'];
          that.globals.optimalNumberOfNeighbors = values['neighbors'];
        })
      }

      this.e6 = 4;
    },
    handleNext4() {
      if (typeof eel !== 'undefined') {
        // eslint-disable-next-line no-undef
        eel.set_optimal_neighbors(this.globals.optimalNumberOfNeighbors);
      }

      this.e6 = 5;
    },
    handleNext5() {
      if (typeof eel !== 'undefined') {
        // eslint-disable-next-line no-undef
        eel.set_features(this.features);
      }

      this.e6 = 6;
    },
    handleNameChange() {
      this.conditionName = this.conditionName.replace(/[^a-zA-Z0-9]+/gi, '');
      if (this.conditionName === '' || this.conditionName === this.condition2Name || this.conditionName === this.condition3Name || this.conditionName === this.condition4Name)
        this.conditionName = 'Label1';
    },
    handleNameChange2() {
      this.condition2Name = this.condition2Name.replace(/[^a-zA-Z0-9]+/gi, '');
      if (this.condition2Name === '' || this.condition2Name === this.conditionName || this.condition2Name === this.condition3Name || this.condition2Name === this.condition4Name)
        this.condition2Name = 'Label2';
    },
    handleNameChange3() {
      this.condition3Name = this.condition3Name.replace(/[^a-zA-Z0-9]+/gi, '');
      if (this.condition3Name === '' || this.condition3Name === this.conditionName || this.condition3Name === this.condition2Name || this.conditionName === this.condition4Name)
        this.condition3Name = 'Label3';
    },
    handleNameChange4() {
      this.condition4Name = this.condition4Name.replace(/[^a-zA-Z0-9]+/gi, '');
      if (this.condition4Name === '' || this.condition4Name === this.condition2Name || this.condition4Name === this.condition3Name || this.condition4Name === this.conditionName)
        this.condition4Name = 'Label4';
    },
    selectInputFolder() {
      if (typeof eel === 'undefined') {
        this.fileInputValue = new File(["dummy"], "dummy/path");
        return;
      }

      var that = this;
      // eslint-disable-next-line no-undef
      eel.select_path_1()(function (path) {
        that.path = path
        that.fileInputValue = new File(["folder"], path);
      })
    },
    selectInputFolder2() {
      if (typeof eel === 'undefined') {
        this.fileInput2Value = new File(["dummy"], "dummy/path");
        return;
      }

      var that = this;
      // eslint-disable-next-line no-undef
      eel.select_path_2()(function (path) {
        that.path2 = path
        that.fileInput2Value = new File(["folder"], path);
      })
    },
    selectInputFolder3() {
      if (typeof eel === 'undefined') {
        this.fileInput3Value = new File(["dummy"], "dummy/path");
        return;
      }

      var that = this;
      // eslint-disable-next-line no-undef
      eel.select_path_3()(function (path) {
        that.path3 = path
        that.fileInput3Value = new File(["folder"], path);
      })
    },
    selectInputFolder4() {
      if (typeof eel === 'undefined') {
        this.fileInput4Value = new File(["dummy"], "dummy/path");
        return;
      }

      var that = this;
      // eslint-disable-next-line no-undef
      eel.select_path_4()(function (path) {
        that.path4 = path
        that.fileInput4Value = new File(["folder"], path);
      })
    },
    selectOutputFolder() {
      if (typeof eel === 'undefined') {
        this.fileOutputValue = new File(["dummy"], "dummy/path");
        return;
      }

      var that = this;
      // eslint-disable-next-line no-undef
      eel.select_output_path()(function (path) {
        that.fileOutputValue = new File(["folder"], path);
      })
    },
    setFeatures(value) {
      if (this.features.find(e => e === value)) {
        this.features = this.features.filter(e => e !== value);
      } else {
        this.features.push(value);
      }

      if (typeof eel === 'undefined') {
        return;
      }
    },
    run() {
      this.done = false;
      this.progress = 0;
      this.progressSteps = 0;
      this.progressCurrentStep = 0;
      this.images.boxplot = [];
      this.images.knn = [];
      this.images.tsne = [];
      this.images.rf = [];

      if (this.e6 === 3) {
        this.e6 = 4;
      }

      if (typeof eel === 'undefined') {
        return;
      }

      this.report = "";

      let that = this;
      // eslint-disable-next-line no-undef
      eel.run_different_nds()(function () {
        that.done = true;
      });
    },
    // cancel() {
    //   if (typeof eel === 'undefined' || this.progressCurrentStep === this.progressSteps) {
    //     return;
    //   }
    //
    //   this.done = true;
    //
    //   // eslint-disable-next-line no-undef
    //   eel.cancel_different_nds();
    // },
  },
  mounted() {
    if (typeof eel === 'undefined') {
      return;
    }

    function addImageUnbound(image, p, i, index) {
      image.show = false;
      this.progress = (100 / p) * i;
      this.progressCurrentStep = i;
      this.progressSteps = p;

      return this.images[index].push(image);
    }
    let addImage = addImageUnbound.bind(this);

    // eslint-disable-next-line no-undef
    eel.expose(addImage, 'add_image_different_nd');

    function printToReportUnbound(text) {
      this.report += text.replace(/\n/g, "<br />") + '<br />------------------------------------------------<br />';
    }
    let printToReport = printToReportUnbound.bind(this);

    // eslint-disable-next-line no-undef
    eel.expose(printToReport, 'print_to_report');
  }
}
</script>
