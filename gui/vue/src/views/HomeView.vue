<template>
  <div>
    <div v-if="!sameNDs && !differentNDs">
      <v-container>
        <v-row>
          <v-col cols="12">
            <h1>Which way do you want to go?</h1>
            <v-row>
              <v-col cols="12">
                <button class="btn-primary mr-3" @click="sameNDs=true">Segmentation and feature extraction of same NDs</button>
                <button class="btn-primary mr-3" @click="differentNDs=true">Comparison between different NDs</button>
              </v-col>
            </v-row>
          </v-col>
        </v-row>
      </v-container>
    </div>
    <div>
      <SameNDs v-if="sameNDs" class="mt-5"/>
      <DifferentNDs v-if="differentNDs" class="mt-5"/>
    </div>
  </div>
</template>

<script>
import SameNDs from '../components/SameNDs'
import DifferentNDs from '../components/DifferentNDs'

export default {
  name: 'HomeView',
  data: () => {
    return {
      sameNDs: false,
      differentNDs: false,
    }
  },
  components: {
    SameNDs,
    DifferentNDs,
  },
}
</script>
