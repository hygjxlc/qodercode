<template>
  <div class="conference-records">
    <el-page-header @back="goBack" :title="`会务历史 - ${expertName}`" />
    
    <el-card class="records-card">
      <template #header>
        <div class="card-header">
          <span>历史会务参与</span>
          <el-button type="primary" @click="showAddDialog">
            <el-icon><Plus /></el-icon>添加记录
          </el-button>
        </div>
      </template>
      
      <el-table :data="records" v-loading="loading" stripe>
        <el-table-column prop="conferenceName" label="会议名称" width="200" />
        <el-table-column prop="conferenceDate" label="会议日期" width="120">
          <template #default="scope">
            {{ formatDate(scope.row.conferenceDate) }}
          </template>
        </el-table-column>
        <el-table-column prop="role" label="参与角色" width="120" />
        <el-table-column prop="topic" label="演讲主题" width="200" show-overflow-tooltip />
        <el-table-column prop="feedback" label="专家反馈" width="200" show-overflow-tooltip />
        <el-table-column label="状态" width="100">
          <template #default="scope">
            <el-tag :type="scope.row.isCompleted ? 'success' : 'info'">
              {{ scope.row.isCompleted ? '已完成' : '待反馈' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150" fixed="right">
          <template #default="scope">
            <el-button 
              v-if="!scope.row.isCompleted" 
              size="small" 
              type="primary"
              @click="addFeedback(scope.row)"
            >
              添加反馈
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <el-empty v-if="records.length === 0 && !loading" description="暂无会务记录" />
    </el-card>
    
    <!-- 添加会务记录对话框 -->
    <el-dialog v-model="addDialogVisible" title="添加会务记录" width="500px">
      <el-form :model="addForm" :rules="addRules" ref="addFormRef" label-width="100px">
        <el-form-item label="会议名称" prop="conferenceName">
          <el-input v-model="addForm.conferenceName" />
        </el-form-item>
        <el-form-item label="会议日期" prop="conferenceDate">
          <el-date-picker
            v-model="addForm.conferenceDate"
            type="date"
            placeholder="选择日期"
            style="width: 100%"
          />
        </el-form-item>
        <el-form-item label="参与角色" prop="role">
          <el-select v-model="addForm.role" style="width: 100%">
            <el-option label="演讲嘉宾" value="演讲嘉宾" />
            <el-option label="主持人" value="主持人" />
            <el-option label="参会者" value="参会者" />
          </el-select>
        </el-form-item>
        <el-form-item label="演讲主题" prop="topic">
          <el-input v-model="addForm.topic" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="addDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitAddForm">确定</el-button>
      </template>
    </el-dialog>
    
    <!-- 添加反馈对话框 -->
    <el-dialog v-model="feedbackDialogVisible" title="添加反馈" width="500px">
      <el-form :model="feedbackForm" :rules="feedbackRules" ref="feedbackFormRef" label-width="100px">
        <el-form-item label="反馈内容" prop="feedback">
          <el-input
            v-model="feedbackForm.feedback"
            type="textarea"
            :rows="4"
            placeholder="请输入专家反馈"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="feedbackDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="submitFeedback">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script>
import { ref, reactive, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { conferenceApi, expertApi } from '../api/expert'

export default {
  name: 'ConferenceRecords',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const loading = ref(false)
    const records = ref([])
    const expertName = ref('')
    const addDialogVisible = ref(false)
    const feedbackDialogVisible = ref(false)
    const addFormRef = ref(null)
    const feedbackFormRef = ref(null)
    const currentRecordId = ref(null)
    
    const addForm = reactive({
      expertId: null,
      conferenceName: '',
      conferenceDate: null,
      role: '',
      topic: ''
    })
    
    const feedbackForm = reactive({
      feedback: ''
    })
    
    const addRules = {
      conferenceName: [{ required: true, message: '请输入会议名称', trigger: 'blur' }],
      conferenceDate: [{ required: true, message: '请选择会议日期', trigger: 'change' }]
    }
    
    const feedbackRules = {
      feedback: [{ required: true, message: '请输入反馈内容', trigger: 'blur' }]
    }
    
    const fetchExpert = async () => {
      try {
        const res = await expertApi.getExpertById(route.params.id)
        if (res.data.success) {
          expertName.value = res.data.data.name
        }
      } catch (error) {
        ElMessage.error('获取专家信息失败')
      }
    }
    
    const fetchRecords = async () => {
      loading.value = true
      try {
        const res = await conferenceApi.getConferenceRecords(route.params.id)
        if (res.data.success) {
          records.value = res.data.data
        }
      } catch (error) {
        ElMessage.error('获取会务记录失败')
      } finally {
        loading.value = false
      }
    }
    
    const goBack = () => {
      router.push(`/expert/${route.params.id}`)
    }
    
    const showAddDialog = () => {
      addForm.expertId = parseInt(route.params.id)
      addForm.conferenceName = ''
      addForm.conferenceDate = null
      addForm.role = ''
      addForm.topic = ''
      addDialogVisible.value = true
    }
    
    const submitAddForm = async () => {
      const valid = await addFormRef.value.validate().catch(() => false)
      if (!valid) return
      
      try {
        await conferenceApi.addConferenceRecord(addForm)
        ElMessage.success('添加成功')
        addDialogVisible.value = false
        fetchRecords()
      } catch (error) {
        ElMessage.error(error.response?.data?.message || '添加失败')
      }
    }
    
    const addFeedback = (row) => {
      currentRecordId.value = row.id
      feedbackForm.feedback = ''
      feedbackDialogVisible.value = true
    }
    
    const submitFeedback = async () => {
      const valid = await feedbackFormRef.value.validate().catch(() => false)
      if (!valid) return
      
      try {
        await conferenceApi.addFeedback(currentRecordId.value, feedbackForm.feedback)
        ElMessage.success('反馈添加成功')
        feedbackDialogVisible.value = false
        fetchRecords()
      } catch (error) {
        ElMessage.error(error.response?.data?.message || '添加失败')
      }
    }
    
    const formatDate = (date) => {
      if (!date) return '-'
      return new Date(date).toLocaleDateString('zh-CN')
    }
    
    onMounted(() => {
      fetchExpert()
      fetchRecords()
    })
    
    return {
      loading,
      records,
      expertName,
      addDialogVisible,
      feedbackDialogVisible,
      addForm,
      feedbackForm,
      addFormRef,
      feedbackFormRef,
      addRules,
      feedbackRules,
      goBack,
      showAddDialog,
      submitAddForm,
      addFeedback,
      submitFeedback,
      formatDate
    }
  }
}
</script>

<style scoped>
.conference-records {
  max-width: 1200px;
  margin: 0 auto;
}

.records-card {
  margin-top: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>
