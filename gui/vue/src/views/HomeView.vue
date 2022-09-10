<template>
  <div>
    <div v-if="!sameNDs && !differentNDs">
      <v-container>
        <v-row>
          <v-col cols="12">
            <h2>Say hello to science!</h2>
            <v-btn small color="primary" @click="print">
                  print
                </v-btn>
            <p class="text-sm-body-2">
              Nanodomain'em all night long, baby! If you had enough of doing lame normie
              stuff, try using this state of the art compooter thingy. Solving problems you didnt know u had in seconds.
              Also we provide you with the latest lorem ipsum.


            </p>
            <v-row>
              <v-col cols="6">
                <h3>Same NDs</h3>
                <p class="text-sm-body-2">
                  Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut
                  labore
                  et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea
                  rebum.
                </p>
                <v-btn small color="primary" @click="sameNDs=true">
                  Start
                </v-btn>
              </v-col>

              <v-col cols="6">
                <h3>Same but different NDs</h3>
                <p class="text-sm-body-2">
                  Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut
                  labore
                  et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea
                  rebum.
                </p>
                <v-btn small color="primary" @click="differentNDs=true">
                  Start
                </v-btn>
              </v-col>
            </v-row>
          </v-col>
        </v-row>
      </v-container>
    </div>
    <div>
      <SameNDs v-if="sameNDs" @back="sameNDs=false" class="mt-5"/>
      <DifferentNDs v-if="differentNDs" @back="differentNDs=false" class="mt-5"/>
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
  methods: {
    print() {
      // eslint-disable-next-line no-undef
      eel.print_object({
        test: 'abc',
        funk: 1,
        funky: 1.2,
        bool: true
      });
    }
  },
  components: {
    SameNDs,
    DifferentNDs,
  },
}
</script>
