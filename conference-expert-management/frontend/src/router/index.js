import { createRouter, createWebHistory } from 'vue-router'
import Layout from '../views/Layout.vue'
import ExpertList from '../views/ExpertList.vue'
import ExpertDetail from '../views/ExpertDetail.vue'
import ContactRecords from '../views/ContactRecords.vue'
import ConferenceRecords from '../views/ConferenceRecords.vue'
import Statistics from '../views/Statistics.vue'

const routes = [
  {
    path: '/',
    component: Layout,
    children: [
      {
        path: '',
        name: 'ExpertList',
        component: ExpertList
      },
      {
        path: 'expert/:id',
        name: 'ExpertDetail',
        component: ExpertDetail
      },
      {
        path: 'expert/:id/contacts',
        name: 'ContactRecords',
        component: ContactRecords
      },
      {
        path: 'expert/:id/conferences',
        name: 'ConferenceRecords',
        component: ConferenceRecords
      },
      {
        path: 'statistics',
        name: 'Statistics',
        component: Statistics
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
